// GLM-5.3-Flash — W5b-2: the decoder layer and the assembled text forward.
// See `glm5_next_layer.h` for the oracle, the port anchors and the manifold.
#include "vllm/model_executor/models/glm5_next_layer.h"

#include "vllm/model_executor/models/glm5_next_diag.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace vllm::glm5_next {
namespace {

[[noreturn]] void Fail(const std::string& what) {
  throw std::runtime_error("glm5_next layer: " + what);
}

void RequireSize(const char* what, size_t got, int64_t want) {
  if (static_cast<int64_t>(got) != want) {
    Fail(std::string(what) + " must hold " + std::to_string(want) +
         " floats, got " + std::to_string(got));
  }
}

// `Glm5NextTextRMSNorm.forward` (`:75-80`): variance over the LAST axis, the eps
// INSIDE the rsqrt, the gain after, and the whole thing in f32 — upstream's own
// `.to(torch.float32)` at `:77`, not a widening this port chose.
void RmsNorm(const float* in, const float* gamma, int64_t n, double eps,
             float* out) {
  double acc = 0.0;
  for (int64_t i = 0; i < n; ++i) acc += static_cast<double>(in[i]) * in[i];
  const float inv =
      static_cast<float>(1.0 / std::sqrt(acc / static_cast<double>(n) + eps));
  for (int64_t i = 0; i < n; ++i) out[i] = gamma[i] * (in[i] * inv);
}

}  // namespace

// See the header: public so the weight bridge and the forward resolve ONE
// geometry rather than two.
glm5_next_kda::Glm5NextKdaDims KdaDimsFrom(const Glm5NextParams& p) {
  glm5_next_kda::Glm5NextKdaDims d;
  d.hidden_size = p.hidden_size;
  d.num_heads = p.kda.num_heads;
  d.head_dim = p.kda.head_dim;
  d.conv_kernel_size = p.kda.conv_kernel_dim;
  d.rms_norm_eps = p.rms_norm_eps;
  d.gate_lower_bound = p.kda.lower_bound;
  return d;
}

std::vector<float> ExpandToHiddenStreams(const std::vector<float>& inputs_embeds,
                                         int64_t batch, int64_t seq_len,
                                         int64_t hc_mult, int64_t hidden) {
  if (batch <= 0 || seq_len <= 0 || hc_mult <= 0 || hidden <= 0) {
    Fail("batch, seq_len, hc_mult and hidden_size must all be > 0");
  }
  RequireSize("inputs_embeds", inputs_embeds.size(), batch * seq_len * hidden);
  // `inputs_embeds.unsqueeze(2).expand(-1, -1, hc_mult, -1).contiguous()`
  // (`:1477`). Every stream starts as a COPY of the embedding; they diverge from
  // the first layer's `post`/`comb` fold onward.
  std::vector<float> out(static_cast<size_t>(batch * seq_len * hc_mult * hidden));
  for (int64_t t = 0; t < batch * seq_len; ++t) {
    const float* src = inputs_embeds.data() + t * hidden;
    for (int64_t m = 0; m < hc_mult; ++m) {
      std::copy_n(src, static_cast<size_t>(hidden),
                  out.data() + (t * hc_mult + m) * hidden);
    }
  }
  return out;
}

DecoderLayerResult DecoderLayerForward(
    const Glm5NextParams& p, int64_t layer_idx, const DecoderLayerWeights& w,
    const std::vector<float>& hidden_streams,
    const std::vector<uint8_t>& mask,
    const std::vector<int32_t>* prev_topk_indices, int64_t prev_topk_width,
    int64_t batch, int64_t seq_len, LayerCache* cache, vt::Queue& queue) {
  if (batch <= 0 || seq_len <= 0) Fail("batch and seq_len must be > 0");
  if (layer_idx < 0 || layer_idx >= p.num_hidden_layers) {
    Fail("layer_idx " + std::to_string(layer_idx) + " is outside [0, " +
         std::to_string(p.num_hidden_layers) + ")");
  }
  const int64_t hc = p.mhc.mult;
  const int64_t H = p.hidden_size;
  const int64_t tokens = batch * seq_len;
  if (hc <= 0) Fail("`hc_mult` must be > 0; the residual manifold has no streams");
  RequireSize("hidden_streams", hidden_streams.size(), tokens * hc * H);
  if (static_cast<int64_t>(mask.size()) != tokens) {
    Fail("attention_mask must hold " + std::to_string(tokens) + " entries, got " +
         std::to_string(mask.size()));
  }
  RequireSize("input_layernorm", w.input_layernorm.size(), H);
  RequireSize("post_attention_layernorm", w.post_attention_layernorm.size(), H);

  // The layer's two kinds must agree with the SCHEDULE, not with whichever arm
  // happens to carry values. A layer built from the wrong schedule is the exact
  // "fluent wrong model" this row keeps naming: a KDA arm run where the config
  // says DSA produces finite activations of the right shape.
  if (w.attn_kind != p.layer_types[static_cast<size_t>(layer_idx)]) {
    Fail("layer " + std::to_string(layer_idx) + " carries a " +
         std::string(Glm5NextLayerKindName(w.attn_kind)) +
         " attention arm but `config.layer_types[" + std::to_string(layer_idx) +
         "]` is " +
         std::string(Glm5NextLayerKindName(
             p.layer_types[static_cast<size_t>(layer_idx)])) +
         ". The schedule decides the arm (`:1261-1268`), not the weights.");
  }
  if (w.mlp_kind != p.mlp_layer_types[static_cast<size_t>(layer_idx)]) {
    Fail("layer " + std::to_string(layer_idx) + " carries a " +
         std::string(Glm5NextMlpKindName(w.mlp_kind)) +
         " feed-forward but `config.mlp_layer_types[" +
         std::to_string(layer_idx) + "]` is " +
         std::string(Glm5NextMlpKindName(
             p.mlp_layer_types[static_cast<size_t>(layer_idx)])) +
         ". The schedule decides the block (`:1270-1272`), not the weights.");
  }

  DecoderLayerResult res;
  std::vector<float> streams = hidden_streams;   // `residual` at `:1293`
  std::vector<float> collapsed(static_cast<size_t>(tokens * H));
  std::vector<float> normed(static_cast<size_t>(tokens * H));
  std::vector<deepseek_v4::MhcPreResult> pre(static_cast<size_t>(tokens));

  // ── the attention site's mHC pre (`:1293-1294`) ───────────────────────────
  // ONE `MhcPre` per token: the mapping reads that token's whole `[hc, hidden]`
  // slab and is independent across tokens (`:278`, `F.linear` over the flattened
  // stream axis). `pre` is retained because `:1316-1318` needs the SAME `post`
  // and `comb` this call produced — recomputing them after the sublayer would be
  // arithmetically identical and would double the Sinkhorn cost for nothing.
  for (int64_t t = 0; t < tokens; ++t) {
    std::vector<float> slab(hidden_streams.begin() + t * hc * H,
                            hidden_streams.begin() + (t + 1) * hc * H);
    pre[static_cast<size_t>(t)] =
        MhcPre(slab, w.attn_hc, p.mhc, H, static_cast<float>(p.rms_norm_eps));
    std::copy_n(pre[static_cast<size_t>(t)].layer_input.data(),
                static_cast<size_t>(H), collapsed.data() + t * H);
  }

  if (diag::Level() > 1) {
    const std::string tag = "L" + std::to_string(layer_idx) + " ";
    diag::Stats((tag + "in.streams").c_str(), hidden_streams);
    diag::Stats((tag + "mhc_pre.collapsed").c_str(), collapsed);
  }

  // `self.input_layernorm(hidden_states)` (`:1296`) — applied AFTER the collapse
  // and as a separate module, which is why `glm5_next::MhcPre` deliberately does
  // not fold it (`glm5_next_mhc.h`).
  for (int64_t t = 0; t < tokens; ++t) {
    RmsNorm(collapsed.data() + t * H, w.input_layernorm.data(), H,
            p.rms_norm_eps, normed.data() + t * H);
  }

  if (diag::Level() > 1) {
    diag::Stats(("L" + std::to_string(layer_idx) + " attn_norm.out").c_str(),
                normed);
  }

  // ── the attention arm (`:1297-1315`) ──────────────────────────────────────
  std::vector<float> attn_out;
  if (w.attn_kind == Glm5NextLayerKind::kLinearAttention) {
    // `apply_mask_to_padding_states(hidden_states, attention_mask)` (`:636`):
    // a padded row is ZEROED before the conv and the recurrence, because a
    // recurrent state has no mask to hide behind — a padded row that reached the
    // delta rule would be summed into a state every later token reads.
    for (int64_t t = 0; t < tokens; ++t) {
      if (mask[static_cast<size_t>(t)] != 0) continue;
      std::fill_n(normed.data() + t * H, static_cast<size_t>(H), 0.0F);
    }
    const glm5_next_kda::Glm5NextKdaDims kd = KdaDimsFrom(p);
    if (cache != nullptr && cache->kda.empty()) {
      cache->kda.resize(static_cast<size_t>(batch));
    }
    if (cache != nullptr &&
        static_cast<int64_t>(cache->kda.size()) != batch) {
      Fail("`LayerCache::kda` holds " + std::to_string(cache->kda.size()) +
           " sequence states but the batch is " + std::to_string(batch) +
           "; the KDA recurrence is single-sequence and sharing one state "
           "across requests mixes their histories");
    }
    attn_out.assign(static_cast<size_t>(tokens * H), 0.0F);
    // One sequence at a time: `Glm5NextKdaLayerForward` carries ONE
    // `[heads, head_dim, head_dim]` recurrent state.
    for (int64_t b = 0; b < batch; ++b) {
      const std::vector<float> row(normed.begin() + b * seq_len * H,
                                   normed.begin() + (b + 1) * seq_len * H);
      const std::vector<float> out = glm5_next_kda::Glm5NextKdaLayerForward(
          w.kda, row, kd, seq_len,
          cache != nullptr ? &cache->kda[static_cast<size_t>(b)] : nullptr,
          queue);
      RequireSize("KDA layer output", out.size(), seq_len * H);
      std::copy_n(out.data(), static_cast<size_t>(seq_len * H),
                  attn_out.data() + b * seq_len * H);
    }
  } else {
    const MlaDims md = MlaDimsFrom(p);
    const IndexerDims idd = IndexerDimsFrom(p);
    const IndexerRole role = IndexerRoleFor(p, layer_idx);
    const IndexerWeights iw = w.dsa.IndexerView();
    // `self.indexer = None if self.skip_topk else Glm5NextTextIndexer(...)`
    // (`:1131`): a `shared` layer passes null, and `Attention` refuses either
    // mismatch by name.
    const AttentionResult a = Attention(
        md, w.dsa.mla, idd, role.skip_topk ? nullptr : &iw, role, normed, mask,
        prev_topk_indices, prev_topk_width, batch, seq_len,
        cache != nullptr ? &cache->dsa : nullptr);
    attn_out = a.attn_output;
    // `topk_indices if self.next_skip_topk else None` (`:1216`). A layer that
    // propagated unconditionally would let a `full` successor be overridden by
    // a stale set it never asked for.
    if (a.propagates_topk) {
      res.topk_indices = a.topk_indices;
      res.topk_width = a.topk_width;
    }
  }
  RequireSize("attention output", attn_out.size(), tokens * H);
  if (diag::Level() > 1) {
    diag::Stats(("L" + std::to_string(layer_idx) + " attn.out (" +
                 std::string(Glm5NextLayerKindName(w.attn_kind)) + ")")
                    .c_str(),
                attn_out);
  }

  // ── the attention site's mHC post (`:1316-1318`) ──────────────────────────
  // `post.unsqueeze(-1) * hidden.unsqueeze(-2) + matmul(comb.transpose(-1,-2),
  // residual)` — the fold that puts the sublayer's single output BACK onto the
  // four streams. This is the line an early-collapsing port has nowhere to put.
  for (int64_t t = 0; t < tokens; ++t) {
    const std::vector<float> out(attn_out.begin() + t * H,
                                 attn_out.begin() + (t + 1) * H);
    const std::vector<float> resid(streams.begin() + t * hc * H,
                                   streams.begin() + (t + 1) * hc * H);
    const std::vector<float> mixed =
        MhcPost(out, resid, pre[static_cast<size_t>(t)], hc, H);
    RequireSize("mHC attention fold", mixed.size(), hc * H);
    std::copy_n(mixed.data(), static_cast<size_t>(hc * H),
                streams.data() + t * hc * H);
  }

  if (diag::Level() > 1) {
    diag::Stats(("L" + std::to_string(layer_idx) + " streams@attn_fold").c_str(),
                streams);
  }

  // ── the feed-forward site's mHC pre (`:1320-1323`) ────────────────────────
  // `residual = hidden_states` (`:1320`) is the manifold the ATTENTION fold just
  // produced, not the layer's input. The two sites therefore have different
  // residuals and the same structure.
  const std::vector<float> ffn_residual = streams;
  for (int64_t t = 0; t < tokens; ++t) {
    const std::vector<float> slab(ffn_residual.begin() + t * hc * H,
                                  ffn_residual.begin() + (t + 1) * hc * H);
    pre[static_cast<size_t>(t)] =
        MhcPre(slab, w.ffn_hc, p.mhc, H, static_cast<float>(p.rms_norm_eps));
    RmsNorm(pre[static_cast<size_t>(t)].layer_input.data(),
            w.post_attention_layernorm.data(), H, p.rms_norm_eps,
            normed.data() + t * H);
  }

  // ── the feed-forward arm (`:1324`) ────────────────────────────────────────
  std::vector<float> mlp_out;
  if (w.mlp_kind == Glm5NextMlpKind::kDense) {
    mlp_out = DenseMlpForward(w.dense_mlp, normed, H, p.intermediate_size, tokens,
                              static_cast<float>(p.swiglu_limit));
  } else {
    mlp_out = MoeForward(MoeDimsFrom(p), w.moe, normed, tokens, queue);
  }
  RequireSize("feed-forward output", mlp_out.size(), tokens * H);
  if (diag::Level() > 1) {
    diag::Stats(("L" + std::to_string(layer_idx) + " ffn_norm.out").c_str(),
                normed);
    diag::Stats(("L" + std::to_string(layer_idx) + " mlp.out (" +
                 std::string(Glm5NextMlpKindName(w.mlp_kind)) + ")")
                    .c_str(),
                mlp_out);
  }

  // ── the feed-forward site's mHC post (`:1325-1327`) ───────────────────────
  for (int64_t t = 0; t < tokens; ++t) {
    const std::vector<float> out(mlp_out.begin() + t * H,
                                 mlp_out.begin() + (t + 1) * H);
    const std::vector<float> resid(ffn_residual.begin() + t * hc * H,
                                   ffn_residual.begin() + (t + 1) * hc * H);
    const std::vector<float> mixed =
        MhcPost(out, resid, pre[static_cast<size_t>(t)], hc, H);
    std::copy_n(mixed.data(), static_cast<size_t>(hc * H),
                streams.data() + t * hc * H);
  }

  res.hidden_streams = std::move(streams);
  return res;
}

namespace {

// The resident tower as a `LayerWeightSource`. It holds nothing of its own and
// hands back a reference into the caller's `TextModelWeights`, so the resident
// overload below is a delegation and not a copy.
class ResidentLayerSource final : public LayerWeightSource {
 public:
  explicit ResidentLayerSource(const TextModelWeights& w) : w_(&w) {}
  int64_t size() const override { return static_cast<int64_t>(w_->layers.size()); }
  const DecoderLayerWeights& Layer(int64_t i) override {
    return w_->layers[static_cast<size_t>(i)];
  }

 private:
  const TextModelWeights* w_;
};

}  // namespace

std::vector<float> TextModelForward(const TextModelWeights& w,
                                    const std::vector<float>& inputs_embeds,
                                    const std::vector<uint8_t>& mask,
                                    int64_t batch, int64_t seq_len,
                                    std::vector<LayerCache>* caches,
                                    vt::Queue& queue) {
  ResidentLayerSource src(w);
  return TextModelForward(w.params, w.norm, src, inputs_embeds, mask, batch,
                          seq_len, caches, queue);
}

std::vector<float> TextModelForward(const Glm5NextParams& p,
                                    const std::vector<float>& norm,
                                    LayerWeightSource& layers,
                                    const std::vector<float>& inputs_embeds,
                                    const std::vector<uint8_t>& mask,
                                    int64_t batch, int64_t seq_len,
                                    std::vector<LayerCache>* caches,
                                    vt::Queue& queue) {
  const int64_t hc = p.mhc.mult;
  const int64_t H = p.hidden_size;
  const int64_t tokens = batch * seq_len;
  if (batch <= 0 || seq_len <= 0) Fail("batch and seq_len must be > 0");
  RequireSize("inputs_embeds", inputs_embeds.size(), tokens * H);
  if (static_cast<int64_t>(mask.size()) != tokens) {
    Fail("attention_mask must hold " + std::to_string(tokens) + " entries, got " +
         std::to_string(mask.size()));
  }
  RequireSize("model.norm", norm.size(), H);

  // `self.layers[: self.config.num_hidden_layers]` (`:1480`), checked rather
  // than sliced. `blk.45` of the published artifact is the MTP block and is NOT
  // a decoder layer: building it as a 46th would be a fluent wrong model that no
  // gate on this fleet could detect (`glm5_next_layer.h`, `glm5_next_loader.h`).
  if (layers.size() != p.num_hidden_layers) {
    Fail("the weight tower holds " + std::to_string(layers.size()) +
         " decoder layers but `num_hidden_layers` is " +
         std::to_string(p.num_hidden_layers) +
         ". On the published checkpoint the 46th block (`blk.45`) is the "
         "multi-token-prediction head and is not a decoder layer; see "
         ".agents/specs/glm5-next-flash.md and issue #2241.");
  }
  if (caches != nullptr &&
      static_cast<int64_t>(caches->size()) != p.num_hidden_layers) {
    Fail("the cache holds " + std::to_string(caches->size()) +
         " layer states but the model has " +
         std::to_string(p.num_hidden_layers) + " layers");
  }

  std::vector<float> streams =
      ExpandToHiddenStreams(inputs_embeds, batch, seq_len, hc, H);

  // `topk_indices = None` (`:1479`) and then the PREVIOUS layer's return
  // (`:1489`). A KDA layer returns nothing, so it WIPES the thread — carrying
  // the last DSA selection across it would feed a `shared` layer a stale key set
  // instead of letting `Attention` refuse.
  std::vector<int32_t> topk;
  int64_t topk_width = 0;
  for (int64_t i = 0; i < p.num_hidden_layers; ++i) {
    // `Layer(i)` is valid only until the next call, so the layer's weights are
    // consumed HERE and never retained — which is what makes a streaming source
    // hold one layer rather than every layer visited so far.
    DecoderLayerResult r = DecoderLayerForward(
        p, i, layers.Layer(i), streams, mask,
        topk.empty() ? nullptr : &topk, topk_width, batch, seq_len,
        caches != nullptr ? &(*caches)[static_cast<size_t>(i)] : nullptr, queue);
    streams = std::move(r.hidden_streams);
    topk = std::move(r.topk_indices);
    topk_width = topk.empty() ? 0 : r.topk_width;
    // ONE LINE PER LAYER, which is what makes this a bisect rather than a
    // verdict: the first layer whose streams stop being finite, or stop
    // varying, is the layer that owns the defect. A single reading of the last
    // layer says the model is broken and nothing about where.
    if (diag::Level() > 0) {
      diag::Stats(("layer[" + std::to_string(i) + "].streams out").c_str(),
                  streams);
    }
  }

  // `self.norm(self.hc_head(hidden_states))` (`:1493`). `HcHeadCollapseMean` is
  // an UNWEIGHTED mean and NOT `deepseek_v4::HcHeadCollapse`: substituting V4's
  // weighted collapse returns `(2 + 4e-6)x` the mean at `hc_mult == 4` and emits
  // fluent text through a final projection that is off by a factor of two
  // (`glm5_next_mhc.h`).
  std::vector<float> out(static_cast<size_t>(tokens * H));
  for (int64_t t = 0; t < tokens; ++t) {
    const std::vector<float> slab(streams.begin() + t * hc * H,
                                  streams.begin() + (t + 1) * hc * H);
    const std::vector<float> collapsed = HcHeadCollapseMean(slab, hc, H);
    if (diag::Level() > 0 && t == tokens - 1) {
      diag::Stats("hc_head collapse (last token)", collapsed);
    }
    RmsNorm(collapsed.data(), norm.data(), H, p.rms_norm_eps,
            out.data() + t * H);
  }
  return out;
}

}  // namespace vllm::glm5_next
