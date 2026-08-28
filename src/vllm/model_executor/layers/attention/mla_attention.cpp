// The MLA attention block + weight absorption — MLA campaign W6.
// Header (include/vllm/model_executor/models/mla_attention.h) carries the full
// `file:line`-on-both-sides port map; this TU implements it.
#include "vllm/model_executor/models/mla_attention.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <numbers>
#include <stdexcept>
#include <vector>

#include "vt/dtype.h"
#include "vt/op_provider.h"

namespace vllm {
namespace mla {
namespace {

using dense_attn::DBuf;
using dense_attn::Dev;
using dense_attn::MakeTensor;
using dense_attn::Reshape;
using vt::DType;
using vt::Tensor;

// A strided rank-3 view: `base` reinterpreted with an explicit element offset,
// shape and stride. Used for every place upstream takes a `.transpose(0,1)` or a
// trailing-column slice — the MLA block is full of them (the rope slice of q,
// the (N,B,·) bmm operands, the nope/v halves of the kv_b_proj output).
Tensor View3(const Tensor& base, int64_t elem_offset, int64_t d0, int64_t d1, int64_t d2,
             int64_t s0, int64_t s1, int64_t s2) {
  Tensor t;
  t.data = static_cast<char*>(base.data) +
           static_cast<size_t>(elem_offset) * vt::SizeOf(base.dtype);
  t.dtype = base.dtype;
  t.device = base.device;
  t.rank = 3;
  t.shape[0] = d0;
  t.shape[1] = d1;
  t.shape[2] = d2;
  t.stride[0] = s0;
  t.stride[1] = s1;
  t.stride[2] = s2;
  return t;
}

Tensor View2(const Tensor& base, int64_t elem_offset, int64_t d0, int64_t d1, int64_t s0) {
  Tensor t;
  t.data = static_cast<char*>(base.data) +
           static_cast<size_t>(elem_offset) * vt::SizeOf(base.dtype);
  t.dtype = base.dtype;
  t.device = base.device;
  t.rank = 2;
  t.shape[0] = d0;
  t.shape[1] = d1;
  t.stride[0] = s0;
  t.stride[1] = 1;
  return t;
}

void RequireWeight(const Tensor& t, const char* name) {
  if (t.data == nullptr) {
    throw std::invalid_argument(std::string("MLA block: required weight `") + name +
                                "` is not set for the selected q_lora_rank branch");
  }
}

// Tier-A2+A5 fold: merge the kv_c+k_pe A-projections into ONE vt::MatmulBT over the
// merged [L+R, H] weight and fuse the {kv_a_layernorm ; decoupled-k_pe RoPE} pair
// into ONE vt::FusedNormRope launch. Default-ON; VT_MLA_FUSED_NORM_ROPE=0 rolls back
// to the byte-exact split path (3/2 A-proj GEMMs + standalone RmsNorm + RopeFromCache).
bool MlaFusedNormRopeEnabled() {
  const char* v = std::getenv("VT_MLA_FUSED_NORM_ROPE");
  return v == nullptr || v[0] != '0';
}

// `yarn_find_correction_dim` (rotary_embedding/common.py:34-42).
double YarnFindCorrectionDim(double num_rotations, int64_t dim, double base,
                             int64_t max_position_embeddings) {
  return (static_cast<double>(dim) *
          std::log(static_cast<double>(max_position_embeddings) /
                   (num_rotations * 2.0 * std::numbers::pi_v<double>))) /
         (2.0 * std::log(base));
}

}  // namespace

void MlaBlockDims::Validate() const {
  if (hidden_size <= 0 || num_heads <= 0 || qk_nope_head_dim <= 0 || v_head_dim <= 0 ||
      kv_lora_rank <= 0) {
    throw std::invalid_argument("MlaBlockDims: every dimension must be > 0");
  }
  if (q_lora_rank < 0) throw std::invalid_argument("MlaBlockDims: q_lora_rank must be >= 0");
  // ─── NoPE (GLM-5.3-Flash, W3, #2213) ─────────────────────────────────────
  // `qk_rope_head_dim == 0` is the ABSENT state of the decoupled rotary, not an
  // invalid width. `Glm5NextTextConfig.validate_architecture` REQUIRES it —
  // "Expecting NoPE for the DSA attention layers, but got {n} as RoPE dim."
  // (configuration_glm5_next.py:225-227) — so an `> 0` rule here and that rule
  // upstream are exact complements and no value satisfies both. At 0 there is
  // no rope slice in the cache row, `head_size()` collapses to `kv_lora_rank`
  // (512, not 576), `qk_head_dim()` is the nope part alone, and every rope
  // branch in this file becomes NOT TAKEN rather than zero-width work.
  //
  // Kimi-Linear is the near miss and is NOT this: `mla_use_nope = true` with
  // `qk_rope_head_dim = 64` keeps the 576-wide row and skips only the rotation
  // (kimi_linear.h:86-88). Its geometry is untouched by this clause.
  //
  // NEGATIVE is still refused, and by its own message: a negative width is a
  // caller that computed a slice and got the sign wrong, which would otherwise
  // reach `View3` as a shape.
  if (qk_rope_head_dim < 0) {
    throw std::invalid_argument(
        "MlaBlockDims: qk_rope_head_dim must be >= 0 (0 is the NoPE state — "
        "GLM-5.3-Flash's `validate_architecture` REQUIRES it; a rope-bearing "
        "layer is > 0 and even)");
  }
  if (qk_rope_head_dim % 2 != 0) {
    throw std::invalid_argument(
        "MlaBlockDims: qk_rope_head_dim must be even (it is the ROTARY dim; "
        "deepseek_v2.py:1059-1064 builds the rope over qk_rope_head_dim only)");
  }
  // A rotation STYLE on a layer with no rotation. Upstream builds no rotary for
  // this model at all — `Glm5NextTextConfig.__post_init__` deletes the inherited
  // `rope_parameters` and the layer is handed `position_embeddings=None` — so
  // there is no cos/sin pairing for either flag to select, and a set flag is a
  // caller that believes it is on a DeepSeek layer. Refuse rather than ignore.
  if (qk_rope_head_dim == 0 && (is_neox_style || indexer_rope_is_neox_style)) {
    throw std::invalid_argument(
        "MlaBlockDims: qk_rope_head_dim is 0 (NoPE), so there is no rotation to "
        "style — `is_neox_style` / `indexer_rope_is_neox_style` describe a "
        "cos/sin application pairing that this geometry does not have");
  }
  if (v_head_dim > qk_head_dim()) {
    // mla_attention.py / flash_attn.py:164-168 ZERO-PAD V up to the QK width; a
    // wider V has no upstream form.
    throw std::invalid_argument("MlaBlockDims: v_head_dim must be <= qk_head_dim");
  }
  if (scale <= 0.0f) {
    throw std::invalid_argument(
        "MlaBlockDims: scale must be set via MlaAttentionScale() — it carries the "
        "YaRN mscale^2 correction (deepseek_v2.py:1067-1075)");
  }
  // dots3-note's LoRA rescales (#699). 1.0 is ABSENT; anything else must be a
  // real positive scalar, and `q_lora_scale` must not be set on a geometry that
  // has no q_lora branch to rescale — upstream computes it as
  // `sqrt(hidden_size / q_lora_rank)` (model.py:306), which does not exist when
  // `q_lora_rank` is None.
  if (!(q_lora_scale > 0.0) || !(kv_lora_scale > 0.0)) {
    throw std::invalid_argument(
        "MlaBlockDims: q_lora_scale / kv_lora_scale must be > 0 (1.0 means the "
        "rescale is ABSENT; dots3-note sets sqrt(hidden_size/rank), "
        "model.py:303-307)");
  }
  if (q_lora_scale != 1.0 && !has_q_lora()) {
    throw std::invalid_argument(
        "MlaBlockDims: q_lora_scale is set but q_lora_rank is 0 — there is no "
        "q_a_layernorm output to rescale on the DIRECT q_proj branch "
        "(deepseek_v2.py:1028-1034)");
  }
  // dots3-note's sliding window (#699 W4b-2). 0 is ABSENT; a negative value is
  // a caller that computed `sliding_window - 1` one layer too early, which
  // would otherwise reach the ops as a window that admits nothing.
  if (sliding_window < 0) {
    throw std::invalid_argument(
        "MlaBlockDims: sliding_window must be >= 0 (0 means ABSENT — the full "
        "context; dots3-note's sliding layers set `sliding_window_size` 513, "
        "model.py:456)");
  }
  // dots3-note's DSA indexer (#699 W4b-3c). All three zero is ABSENT; a
  // PARTIAL group is a caller that read some of the config and not the rest,
  // which would otherwise reach the block as a selection over a zero-wide head.
  const int nonzero =
      (index_n_heads > 0 ? 1 : 0) + (index_head_dim > 0 ? 1 : 0) + (index_topk > 0 ? 1 : 0);
  if (nonzero != 0 && nonzero != 3) {
    throw std::invalid_argument(
        "MlaBlockDims: index_n_heads / index_head_dim / index_topk must be set "
        "TOGETHER or left all zero — a partial indexer group cannot describe a "
        "selection (deepseek_v2.py:685-687)");
  }
  if (has_indexer() && index_head_dim < qk_rope_head_dim) {
    throw std::invalid_argument(
        "MlaBlockDims: index_head_dim must be >= qk_rope_head_dim — the indexer "
        "rotates the LEADING qk_rope_head_dim slice of its own head "
        "(deepseek_v2.py:804-806)");
  }
  if (has_indexer() && !has_q_lora()) {
    throw std::invalid_argument(
        "MlaBlockDims: the DSA indexer consumes `q_c`, the q-LoRA latent "
        "(model.py:171-172 passes `q_c` to `attention.indexer`), so it cannot "
        "run on the DIRECT q_proj branch");
  }
}

// mla_attention.py:880-900 + :959-962. Upstream's chain is
//   W = kv_b_proj.weight.T                     [L, N*(P+V)]
//   W = W.view(L, N, P + V)
//   W_UK, W_UV = W.split([P, V], dim=-1)       [L,N,P] , [L,N,V]
//   self.W_UK_T = W_UK.permute(1, 2, 0)        [N,P,L]
//   self.W_UV   = W_UV.transpose(0, 1)         [N,L,V]
// so, folding the transpose into the indices, with `src` the checkpoint-layout
// row-major [N*(P+V), L] weight (torch [out_features, in_features]):
//   W_UK_T[n, p, l] = src[n*(P+V) + p,     l]
//   W_UV  [n, l, v] = src[n*(P+V) + P + v, l]
AbsorbedKvBProj AbsorbKvBProjBf16(const uint16_t* kv_b_proj_weight, const MlaBlockDims& dims) {
  if (kv_b_proj_weight == nullptr) {
    throw std::invalid_argument("AbsorbKvBProjBf16: null kv_b_proj weight");
  }
  dims.Validate();
  const int64_t n = dims.num_heads, p = dims.qk_nope_head_dim;
  const int64_t v = dims.v_head_dim, l = dims.kv_lora_rank;
  const int64_t row = p + v;  // per-head output width (:518-519)
  AbsorbedKvBProj out;
  out.w_uk_t.assign(static_cast<size_t>(n * p * l), 0);
  out.w_uv.assign(static_cast<size_t>(n * l * v), 0);
  for (int64_t h = 0; h < n; ++h) {
    for (int64_t i = 0; i < p; ++i) {
      const uint16_t* src = kv_b_proj_weight + (h * row + i) * l;
      uint16_t* dst = out.w_uk_t.data() + (h * p + i) * l;
      for (int64_t j = 0; j < l; ++j) dst[j] = src[j];
    }
    for (int64_t i = 0; i < v; ++i) {
      const uint16_t* src = kv_b_proj_weight + (h * row + p + i) * l;
      uint16_t* dst = out.w_uv.data() + h * l * v + i;
      for (int64_t j = 0; j < l; ++j) dst[j * v] = src[j];
    }
  }
  return out;
}

// deepseek_scaling_rope.py:20-23.
double YarnGetMscale(double scale, double mscale) {
  if (scale <= 1.0) return 1.0;
  return 0.1 * mscale * std::log(scale) + 1.0;
}

// deepseek_scaling_rope.py:76-118.
std::vector<float> BuildDeepseekRopeCosSinCache(const DeepseekYarnRopeParams& p,
                                                int64_t rows) {
  if (p.rotary_dim <= 0 || p.rotary_dim % 2 != 0) {
    throw std::invalid_argument("BuildDeepseekRopeCosSinCache: rotary_dim must be even and > 0");
  }
  if (rows <= 0) throw std::invalid_argument("BuildDeepseekRopeCosSinCache: rows must be > 0");
  const int64_t rot = p.rotary_dim, half = rot / 2;
  const bool yarn = p.yarn && p.scaling_factor > 1.0;

  // `_compute_inv_freq` (:76-104).
  std::vector<double> inv_freq(static_cast<size_t>(half));
  double low = 0.0, high = 0.0;
  if (yarn) {
    // `yarn_find_correction_range` (common.py:46-59), truncate=True.
    low = std::floor(YarnFindCorrectionDim(p.beta_fast, rot, p.base,
                                           p.original_max_position_embeddings));
    high = std::ceil(YarnFindCorrectionDim(p.beta_slow, rot, p.base,
                                           p.original_max_position_embeddings));
    low = std::max(low, 0.0);
    high = std::min(high, static_cast<double>(rot - 1));
    if (low == high) high += 0.001;  // "Prevent singularity" (common.py:65-66)
  }
  for (int64_t i = 0; i < half; ++i) {
    const double pos_freq = std::pow(p.base, (2.0 * static_cast<double>(i)) /
                                                 static_cast<double>(rot));
    const double extrapolation = 1.0 / pos_freq;
    if (!yarn) {
      inv_freq[static_cast<size_t>(i)] = extrapolation;
      continue;
    }
    const double interpolation = 1.0 / (p.scaling_factor * pos_freq);
    // `yarn_linear_ramp_mask(low, high, rotary_dim // 2, ...)` (common.py:62-70)
    // — note the ramp is over rotary_dim//2 entries while the correction RANGE
    // was computed against rotary_dim; that asymmetry is upstream's and is
    // reproduced rather than "fixed".
    const double linear = (static_cast<double>(i) - low) / (high - low);
    const double ramp = std::min(1.0, std::max(0.0, linear));
    const double mask = (1.0 - ramp) * p.extrapolation_factor;
    inv_freq[static_cast<size_t>(i)] =
        interpolation * (1.0 - mask) + extrapolation * mask;
  }

  // The ROTATION mscale (:55-59) — distinct from the softmax-scale mscale^2.
  const double rot_mscale =
      yarn ? (YarnGetMscale(p.scaling_factor, p.mscale) /
              YarnGetMscale(p.scaling_factor, p.mscale_all_dim) * p.attn_factor)
           : 1.0;

  // `_compute_cos_sin_cache` (:105-118): cache = cat((cos*mscale, sin*mscale)).
  std::vector<float> cache(static_cast<size_t>(rows * rot));
  for (int64_t t = 0; t < rows; ++t) {
    for (int64_t i = 0; i < half; ++i) {
      const double angle = static_cast<double>(t) * inv_freq[static_cast<size_t>(i)];
      cache[static_cast<size_t>(t * rot + i)] =
          static_cast<float>(std::cos(angle) * rot_mscale);
      cache[static_cast<size_t>(t * rot + half + i)] =
          static_cast<float>(std::sin(angle) * rot_mscale);
    }
  }
  return cache;
}

// deepseek_v2.py:995 then :1067-1075.
float MlaAttentionScale(const MlaBlockDims& dims, const DeepseekYarnRopeParams& p) {
  const double base = std::pow(static_cast<double>(dims.qk_head_dim()), -0.5);
  if (!p.yarn || p.scaling_factor <= 1.0) return static_cast<float>(base);
  const double mscale = YarnGetMscale(p.scaling_factor, p.mscale_all_dim);
  return static_cast<float>(base * mscale * mscale);
}

// The `kv_b_proj` up-projection callback W5 left open (mla_chunked_context.h:228
// `MlaUpProjectFn`), i.e. `_compute_prefill_context`'s :2141-2170:
//   kv_c   = workspace[:toks, :kv_lora_rank]
//   k_pe   = workspace[:toks, kv_lora_rank:]
//   kv_nope = kv_b_proj(kv_c).view(-1, N, P + V)
//   k_nope, v = kv_nope.split([P, V], dim=-1)
//   k = _concat_k_nope_k_pe(k_nope, k_pe)
MlaUpProjectFn MakeMlaUpProjectFn(Dev d, const MlaBlockDims& dims, const MlaBlockWeights& w,
                                  MlaUpProjectScratch& scratch) {
  const int64_t n = dims.num_heads, p = dims.qk_nope_head_dim, v = dims.v_head_dim;
  const int64_t dqk = dims.qk_head_dim(), l = dims.kv_lora_rank, r = dims.qk_rope_head_dim;
  const Tensor kv_b = w.kv_b_proj;
  RequireWeight(kv_b, "kv_b_proj");
  MlaUpProjectScratch* sc = &scratch;
  return [d, n, p, v, dqk, l, r, kv_b, sc](vt::Queue& q, const Tensor& ws,
                                           int64_t toks) mutable -> MlaContextChunkKv {
    Dev dd{d.b, q};
    sc->bufs.clear();
    // The workspace row is [kv_lora_rank | qk_rope_head_dim] — the SAME 576-wide
    // layout vt::GatherMlaCache filled, so kv_c and k_pe are column slices.
    Tensor kv_c = View2(ws, 0, toks, l, ws.stride[0]);
    Tensor k_pe = View3(ws, l, toks, 1, r, ws.stride[0], r, 1);
    // `self.kv_b_proj(kv_c)` (:2160) — applied DIRECTLY to the 512-column slice
    // of the 576-wide workspace, exactly as upstream applies F.linear to that
    // view. vt::MatmulBT takes the row-strided activation with no copy (the W6
    // relaxation).
    sc->bufs.emplace_back(dd, ws.dtype, std::vector<int64_t>{toks, n * (p + v)});
    Tensor kv_nope = sc->bufs.back().t();
    vt::MatmulBT(q, kv_nope, kv_c, kv_b);
    // `.view(-1, N, P + V).split([P, V], dim=-1)` — both halves are STRIDED
    // views; no copy.
    Tensor k_nope = View3(kv_nope, 0, toks, n, p, n * (p + v), p + v, 1);
    Tensor value = View3(kv_nope, p, toks, n, v, n * (p + v), p + v, 1);
    sc->bufs.emplace_back(dd, ws.dtype, std::vector<int64_t>{toks, n, dqk});
    Tensor key = sc->bufs.back().t();
    // `_concat_k_nope_k_pe` (:2063-2092): k_pe carries ONE head and is
    // BROADCAST across all N heads.
    vt::ConcatMlaNopeRope(q, key, k_nope, k_pe);
    MlaContextChunkKv out;
    out.k = key;
    out.v = value;
    return out;
  };
}

void ForwardMlaAttentionBlock(Dev d, const MlaBlockDims& dims, const MlaBlockWeights& w,
                              const Tensor& hidden, const Tensor& positions,
                              Tensor& kv_cache, const Tensor& slot_mapping,
                              const MlaBlockMetadata& meta, v1::TritonMLAImpl& impl,
                              Tensor& out) {
  dims.Validate();
  // ─── dots3-note's headwise gate: PRECONDITIONS, checked at ENTRY ─────────
  // Both are properties of the CONFIG and the WEIGHT, knowable before any op
  // runs — and step 4 writes this token's K/V into the paged cache, so a throw
  // from step 5c would leave the cache MUTATED for a request that produced no
  // output. Review finding F6; the checks are the same, only their position
  // moved.
  const bool has_gate = w.attn_gate_proj.data != nullptr;
  if (has_gate && hidden.dtype != DType::kBF16) {
    throw std::invalid_argument(
        "MLA block: the headwise attention gate (`attn_gate_proj`, "
        "model.py:190-197) is realized through vt::SharedExpertGate, which "
        "stores BF16 only — run the block in bf16 or drop the gate. Refusing "
        "rather than narrowing the block or dropping the gate silently.");
  }
  if (has_gate && w.attn_gate_proj.shape[0] != dims.num_heads) {
    throw std::invalid_argument(
        "MLA block: `attn_gate_proj` must be [num_heads, hidden_size] — the "
        "HEADWISE gate has one logit per head (model.py:287-291); the "
        "non-headwise [num_heads*v_head_dim] arm (model.py:198-200) is not "
        "represented here");
  }
  const int64_t T = hidden.shape[0];
  const int64_t H = dims.hidden_size, N = dims.num_heads;
  const int64_t P = dims.qk_nope_head_dim, R = dims.qk_rope_head_dim;
  const int64_t V = dims.v_head_dim, L = dims.kv_lora_rank;
  const int64_t Dqk = dims.qk_head_dim();
  const DType dt = hidden.dtype;
  if (hidden.rank != 2 || hidden.shape[1] != H) {
    throw std::invalid_argument("MLA block: hidden must be [T, hidden_size]");
  }
  if (out.rank != 2 || out.shape[0] != T || out.shape[1] != H) {
    throw std::invalid_argument("MLA block: out must be [T, hidden_size]");
  }
  const int64_t decode_toks = meta.num_decode_tokens;
  if (decode_toks < 0 || decode_toks > T) {
    throw std::invalid_argument(
        "MLA block: num_decode_tokens must be within [0, T] (decode tokens are "
        "packed FIRST — mla_attention.py:700-709)");
  }
  const int64_t prefill_toks = T - decode_toks;
  if (T == 0) return;

  // ─── the SPARSE per-token MQA route (dots3-note W4b-3c, #699) ────────────
  // Both halves of the predicate are load-bearing. `dims.has_indexer()` is a
  // property of the LAYER — dots3-note's full layers carry an indexer and its
  // sliding layers do not (`self.indexer = None`, model.py:432-434). The
  // metadata half is a property of the STEP, and it is upstream's own decision:
  // `use_dense_mha = prefill_max_seq_len <= self.topk_tokens`
  // (sparse_mla_attention.py:296-299 @ bc2d63e650), consumed at
  // mla_attention.py:829-851, which promotes the whole step to MQA
  // (`num_mqa_tokens = q.size(0)`) only when that is false. Below the threshold
  // the top-k selects every causal candidate and dense attention IS upstream's
  // answer, so an empty `indexer_cu_seqlens_q` is mirroring rather than a
  // shortcut — and it is what keeps every gate this seam already passes
  // byte-identical.
  const bool run_indexer = dims.has_indexer() && !meta.indexer_cu_seqlens_q.empty();
  if (run_indexer) {
    if (decode_toks != T) {
      throw std::invalid_argument(
          "MLA block: a sparse step routes EVERY token through MQA "
          "(`num_mqa_tokens = q.size(0)`, mla_attention.py:850), so "
          "num_decode_tokens must equal the token count");
    }
    const std::vector<int32_t>& cu = meta.indexer_cu_seqlens_q;
    if (cu.size() < 2 || cu.front() != 0 || cu.back() != static_cast<int32_t>(T)) {
      throw std::invalid_argument(
          "MLA block: indexer_cu_seqlens_q must be [num_reqs + 1] cumulative "
          "query lengths starting at 0 and ending at the token count");
    }
    for (size_t i = 1; i < cu.size(); ++i) {
      if (cu[i] <= cu[i - 1]) {
        throw std::invalid_argument(
            "MLA block: indexer_cu_seqlens_q must be strictly increasing — an "
            "empty request has no query row for the indexer to select for");
      }
    }
  }

  // ─── 1. the A projections + the query branch (mla.py:126-153) ─────────────
  // DEVIATION (recorded): upstream issues ONE fused GEMM per A-projection module
  // and then `.split(...)`s the result into views. We slice the WEIGHT's output
  // ROWS instead and issue one GEMM per slice, so every downstream consumer gets
  // a CONTIGUOUS buffer — vt::RmsNorm requires contiguous inputs, and relaxing
  // it would touch the hottest op in every existing model for no MLA-specific
  // gain. The checkpoint PACKING is unchanged (`fused_qkv_a_proj` stays one
  // weight, exactly as packed_modules_mapping demands at deepseek_v2.py:
  // 1812-1820) — only the launch granularity differs, which is the same trade
  // the dense block already makes by DEFAULT (dense_attn_block.h's 3-shard qkv
  // path, VT_QWEN3_QKV_MERGE default OFF). A truly fused A-GEMM is a W9 A/B.
  RequireWeight(w.kv_a_layernorm, "kv_a_layernorm");
  // Tier-A2+A5 fold (default-ON). When the shared vt::FusedNormRope op is
  // registered on this backend, the kv_c(nope) + k_pe(rope) A-projections collapse
  // to ONE merged [L+R, H] vt::MatmulBT and the {kv_a_layernorm ; decoupled-k_pe
  // RoPE} pair folds into ONE launch — BIT-IDENTICAL, since the merged GEMM is the
  // same arithmetic with a wider N (per-row output slices unchanged) and the fused
  // op runs the exact {RmsNorm(latent) ; RopeFromCache(k_pe)} the split path does
  // (the two halves are disjoint dims). The latent slice of the merged output is
  // strided, which is why the fused kernel — not vt::RmsNorm, which requires a
  // contiguous input — reads it. VT_MLA_FUSED_NORM_ROPE=0 restores the split path.
  // dots3-note (#699) normalizes the ROPE HALF of the kv row on its own
  // (`k_rope_only_layernorm`, model.py:160) between the A-projection and the
  // rope. vt::FusedNormRope ropes k_pe straight out of the merged [L+R] row and
  // has no step in which that norm could run, so the weight's presence takes
  // the split path. Every DeepSeek registration leaves the weight empty and is
  // therefore unaffected — same predicate, same value, same launches.
  const bool has_k_rope_norm = w.k_rope_only_layernorm.data != nullptr;
  const bool fused_nr = R > 0 && !has_k_rope_norm && MlaFusedNormRopeEnabled() &&
                        vt::OpRegistered(vt::OpId::kFusedNormRope, d.q.device.type);
  DBuf kv_c(d, dt, {T, fused_nr ? int64_t{0} : L});
  DBuf kv_merged(d, dt, {T, fused_nr ? (L + R) : int64_t{0}});
  DBuf k_pe(d, dt, {T, R});
  DBuf q_raw(d, dt, {T, N * Dqk});
  // `q_c`, the q-LoRA latent, is HOISTED out of the branch below because
  // dots3-note's DSA indexer consumes it: `attention.indexer(hidden_states,
  // q_c, positions, attention.indexer_rope_emb)` (model.py:171-172 @
  // `bc2d63e650`). Recomputing it in the model would be a second `q_a_proj`
  // GEMM over a latent this block already produced — the duplicate path
  // AGENTS.md forbids — which is why the indexer lives INSIDE the seam rather
  // than beside it. Zero-width on the direct `q_proj` branch, where there is no
  // latent and `MlaBlockDims::Validate` already refuses an indexer.
  DBuf q_c(d, dt, {T, dims.q_lora_rank});
  Tensor q_c_t = q_c.t();
  if (dims.has_q_lora()) {
    RequireWeight(w.fused_qkv_a_proj, "fused_qkv_a_proj");
    RequireWeight(w.q_a_layernorm, "q_a_layernorm");
    RequireWeight(w.q_b_proj, "q_b_proj");
    const int64_t ql = dims.q_lora_rank;
    const Tensor& fused = w.fused_qkv_a_proj;
    if (fused.shape[0] != ql + L + R) {
      throw std::invalid_argument(
          "MLA block: fused_qkv_a_proj must be [q_lora_rank + kv_lora_rank + "
          "qk_rope_head_dim, hidden_size] (deepseek_v2.py:1004-1009)");
    }
    Tensor w_qa = fused.Slice(0, 0, ql);
    Tensor q_raw_t = q_raw.t();
    vt::MatmulBT(d.q, q_c_t, hidden, w_qa);  // q_c A-proj (own GEMM → contiguous latent)
    if (fused_nr) {
      Tensor kv_merged_t = kv_merged.t();  // A2: ONE merged [T, L+R] kv A-proj GEMM
      vt::MatmulBT(d.q, kv_merged_t, hidden, fused.Slice(0, ql, ql + L + R));
    } else {
      Tensor kv_c_t = kv_c.t(), k_pe_t = k_pe.t();
      vt::MatmulBT(d.q, kv_c_t, hidden, fused.Slice(0, ql, ql + L));
      // NoPE (W3, #2213): with no rope slice there are no rope ROWS in the
      // A-projection either — `fused_qkv_a_proj` is [q_lora + kv_lora, hidden]
      // — so the second GEMM is NOT LAUNCHED rather than issued at width 0,
      // which `Tensor::Slice` refuses as an empty range.
      if (R > 0) {
        vt::MatmulBT(d.q, k_pe_t, hidden, fused.Slice(0, ql + L, ql + L + R));
      }
    }
    // `q_c = self.q_a_layernorm(q_c)` (mla.py:143) — in-place, like upstream.
    vt::RmsNorm(d.q, q_c_t, q_c_t, w.q_a_layernorm, vt::RmsNormArgs{dims.rms_norm_eps, false});
    // dots3-note: `* q_lora_scale`, AFTER the layernorm (model.py:155). Not
    // launched at 1.0, which is every DeepSeek registration.
    if (dims.q_lora_scale != 1.0) {
      vt::MulScalar(d.q, q_c_t, q_c_t, dims.q_lora_scale);
    }
    // `q = self.q_b_proj(q_c)[0]` (mla.py:144)
    vt::MatmulBT(d.q, q_raw_t, q_c_t, w.q_b_proj);
  } else {
    RequireWeight(w.kv_a_proj_with_mqa, "kv_a_proj_with_mqa");
    RequireWeight(w.q_proj, "q_proj");
    const Tensor& kva = w.kv_a_proj_with_mqa;
    if (kva.shape[0] != L + R) {
      throw std::invalid_argument(
          "MLA block: kv_a_proj_with_mqa must be [kv_lora_rank + qk_rope_head_dim, "
          "hidden_size] (deepseek_v2.py:511)");
    }
    Tensor q_raw_t = q_raw.t();
    if (fused_nr) {
      Tensor kv_merged_t = kv_merged.t();  // A2: ONE merged GEMM (kva is already [L+R, H])
      vt::MatmulBT(d.q, kv_merged_t, hidden, kva);
    } else {
      Tensor kv_c_t = kv_c.t(), k_pe_t = k_pe.t();
      vt::MatmulBT(d.q, kv_c_t, hidden, kva.Slice(0, 0, L));
      if (R > 0) {  // NoPE: no rope rows to project (W3, #2213)
        vt::MatmulBT(d.q, k_pe_t, hidden, kva.Slice(0, L, L + R));
      }
    }
    // `q = self.q_proj(hidden_states)[0]` (mla.py:152)
    vt::MatmulBT(d.q, q_raw_t, hidden, w.q_proj);
  }

  // ─── 2+3. kv_a_layernorm(LATENT ONLY, deepseek_v2.py:516) + decoupled RoPE
  //          (mla.py:155-167). The decoupled rope part is deliberately NOT normed
  //          — that asymmetry is the whole reason `kv_a_layernorm` is built over
  //          `kv_lora_rank` and not over the full 576-wide projection output.
  //          RoPE rotates only the TRAILING qk_rope_head_dim slice of each query
  //          head; the rotation style comes from `dims.is_neox_style` (DeepSeek-
  //          V2/V3 use the adjacent-pair GPT-J form, is_neox_style=False,
  //          deepseek_v2.py:1059-1064; MiniCPM3 the neox half-split form). All rope
  //          operands are STRIDED views, which is why W6 relaxed vt::RopeFromCache
  //          to stride-driven q/k.
  DBuf kv_c_normed(d, dt, {T, L});
  Tensor kv_c_normed_t = kv_c_normed.t();
  vt::RopeArgs rope;
  rope.rotary_dim = static_cast<int>(R);
  rope.is_neox_style = dims.is_neox_style;
  if (fused_nr) {
    RequireWeight(w.rope_cos_sin_cache, "rope_cos_sin_cache");
    // A5: latent RMSNorm + decoupled k_pe RoPE in ONE launch over the merged kv
    // row; k_pe (roped) written to the same buffer the split path fed the cache.
    Tensor kv_merged_t = kv_merged.t(), k_pe_out = k_pe.t();
    vt::FusedNormRope(d.q, kv_c_normed_t, k_pe_out, kv_merged_t, w.kv_a_layernorm, positions,
                      w.rope_cos_sin_cache, vt::RmsNormArgs{dims.rms_norm_eps, false}, rope);
    // The QUERY rope (q_pe, N heads) stays a distinct binding — bit-identical,
    // since rope is per-head independent (k_pe was roped in the fused op above).
    Tensor q_pe = View3(q_raw.t(), P, T, N, R, N * Dqk, Dqk, 1);
    vt::RopeFromCache(d.q, q_pe, nullptr, positions, w.rope_cos_sin_cache, rope);
  } else {
    Tensor kv_c_in = kv_c.t();
    vt::RmsNorm(d.q, kv_c_normed_t, kv_c_in, w.kv_a_layernorm,
                vt::RmsNormArgs{dims.rms_norm_eps, false});
    // dots3-note: `k_pe = k_rope_only_layernorm(k_pe)` (model.py:160) — an
    // extra RMSNorm over the decoupled-rope slice, BEFORE the rotation. Doing
    // it after the rotation is a different answer and is a mutation the gate
    // fires on; doing it at all is what makes the whole layer invariant to a
    // rescale of the `kv_a_proj_with_mqa` rows that produce k_pe, which
    // DeepSeek — having no such norm — is not.
    if (has_k_rope_norm) {
      Tensor k_pe_t = k_pe.t();
      vt::RmsNorm(d.q, k_pe_t, k_pe_t, w.k_rope_only_layernorm,
                  vt::RmsNormArgs{dims.rms_norm_eps, false});
    }
    if (R > 0) {
      RequireWeight(w.rope_cos_sin_cache, "rope_cos_sin_cache");
      Tensor q_pe = View3(q_raw.t(), P, T, N, R, N * Dqk, Dqk, 1);
      Tensor k_pe3 = View3(k_pe.t(), 0, T, 1, R, R, R, 1);
      vt::RopeFromCache(d.q, q_pe, &k_pe3, positions, w.rope_cos_sin_cache, rope);
    }
  }
  // dots3-note: `* kv_lora_scale`, AFTER kv_a_layernorm (model.py:159) and
  // therefore before the cache write and before every attention read. Not
  // launched at 1.0.
  if (dims.kv_lora_scale != 1.0) {
    vt::MulScalar(d.q, kv_c_normed_t, kv_c_normed_t, dims.kv_lora_scale);
  }

  // ─── 3b. dots3-note's DSA "Lightning Indexer" (model.py:171-172) ─────────
  // Placed exactly where upstream places it: AFTER the decoupled rope and
  // BEFORE `mla_attn`, which is where the cache write happens upstream too.
  // It reads `hidden_states` and `q_c` and produces, per query token, the list
  // of key POSITIONS that token will attend to.
  //
  // ─── WHAT THIS IS A PORT OF, @ `bc2d63e650` ──────────────────────────────
  //   `Indexer.forward`, deepseek_v2.py:751-842, the NON-fused branch
  //   (`:803-842`), which is the one a portable port mirrors — the two fused
  //   branches above it (`:757-802`) are ROCm and CUDA fast paths for the same
  //   arithmetic:
  //     q  = wq_b(q_c).view(-1, n_head, head_dim)               `:754-755`
  //     kw = wk_weights_proj(hidden); k = kw[:, :D]; w = kw[:, D:]  `:807-810`
  //     k  = k_norm(k)                                          `:812`
  //     q_pe, q_nope = split(q, [rope_dim, D - rope_dim])        `:804-806`
  //     k_pe, k_nope = split(k, [rope_dim, D - rope_dim])        `:813-815`
  //     q_pe, k_pe   = rotary_emb(positions, q_pe, k_pe)         `:817`
  //     weights = weights * q_scale * softmax_scale * n_head_scale  `:840`
  //     indexer_op(...)                                          `:842`
  //
  // THREE THINGS A PORT GETS WRONG SILENTLY, and each is a named property here:
  //   * `k_norm` is a `LayerNorm(head_dim, eps=1e-6)` (`:708`) — mean-
  //     subtracting, WITH a bias — not the RmsNorm every other norm on this
  //     model is. It routes through `vt::LayerNorm`, weight and bias both.
  //   * the rope slice is the LEADING `[0, rope_dim)` of the indexer head, not
  //     the trailing one the MAIN MLA rope uses (`:804-806` against
  //     mla.py:200-203, which is where the trailing-slice rope actually is at
  //     this pin — `:160-167` is the `q_lora_rank is not None` assert block, so
  //     that anchor was WRONG under a true claim). It routes through
  //     `vt::RopeFromCache` over a strided leading-slice view.
  //   * the rope PAIRING follows `indexer_rope_interleave` (`:1159`) and is
  //     INDEPENDENT of the main rope's, which dots3-note fixes at GPT-J.
  // Both norms and both ropes are existing gated ops. A second copy of either
  // inside this block would be the parallel path AGENTS.md forbids.
  std::vector<DBuf> ix_bufs;
  Tensor topk_idx{}, topk_cnt{};
  if (run_indexer) {
    const int64_t IH = dims.index_n_heads, ID = dims.index_head_dim;
    const int64_t K = dims.index_topk;
    RequireWeight(w.indexer_wq_b, "indexer_wq_b");
    RequireWeight(w.indexer_wk, "indexer_wk");
    RequireWeight(w.indexer_weights_proj, "indexer_weights_proj");
    RequireWeight(w.indexer_k_norm_weight, "indexer_k_norm_weight");
    RequireWeight(w.indexer_k_norm_bias, "indexer_k_norm_bias");
    RequireWeight(w.rope_cos_sin_cache, "rope_cos_sin_cache");

    ix_bufs.reserve(8);
    ix_bufs.emplace_back(d, dt, std::vector<int64_t>{T, IH * ID});
    Tensor iq = ix_bufs.back().t();
    vt::MatmulBT(d.q, iq, q_c_t, w.indexer_wq_b);  // `:754`
    // TWO GEMMs where upstream issues one merged `wk_weights_proj`
    // (`:700-707`, `:808-810`): the `k` half is handed straight to `k_norm`,
    // and `vt::LayerNorm` requires a CONTIGUOUS input while a merged output's
    // k half is a column slice. Identical arithmetic — a merged GEMM's output
    // rows are independent — and the same launch-count trade the A-projections
    // above already record. The fold is owed to the row.
    ix_bufs.emplace_back(d, dt, std::vector<int64_t>{T, ID});
    Tensor ik_raw = ix_bufs.back().t();
    vt::MatmulBT(d.q, ik_raw, hidden, w.indexer_wk);
    ix_bufs.emplace_back(d, dt, std::vector<int64_t>{T, IH});
    Tensor iw = ix_bufs.back().t();
    vt::MatmulBT(d.q, iw, hidden, w.indexer_weights_proj);
    // `k = self.k_norm(k)` — LayerNorm, weight AND bias, eps the upstream
    // LITERAL 1e-6 (`:708`) rather than the model's `rms_norm_eps`.
    ix_bufs.emplace_back(d, dt, std::vector<int64_t>{T, ID});
    Tensor ik = ix_bufs.back().t();
    vt::LayerNorm(d.q, ik, ik_raw, &w.indexer_k_norm_weight, &w.indexer_k_norm_bias,
                  vt::LayerNormArgs{1e-6f});
    // The LEADING `[0, qk_rope_head_dim)` slice of each indexer head rotates;
    // the indexer's k is MQA, one head per token.
    if (R > 0) {
      vt::RopeArgs irope;
      irope.rotary_dim = static_cast<int>(R);
      irope.is_neox_style = dims.indexer_rope_is_neox_style;
      Tensor iq3 = View3(iq, 0, T, IH, R, IH * ID, ID, 1);
      Tensor ik3 = View3(ik, 0, T, 1, R, ID, ID, 1);
      vt::RopeFromCache(d.q, iq3, &ik3, positions, w.rope_cos_sin_cache, irope);
    }

    ix_bufs.emplace_back(d, DType::kI32, std::vector<int64_t>{T, K});
    topk_idx = ix_bufs.back().t();
    ix_bufs.emplace_back(d, DType::kI32, std::vector<int64_t>{T});
    topk_cnt = ix_bufs.back().t();

    // The candidate range, as two contiguous i32 arrays over the LONGEST
    // request in the step: `win_start` is all zeros and `win_end[i] = i + 1`,
    // so slicing rows `[c0, c1)` of either gives exactly the causal range of
    // those query rows. One pair serves every request and every chunk.
    const std::vector<int32_t>& cu = meta.indexer_cu_seqlens_q;
    int64_t max_len = 1;
    for (size_t i = 1; i < cu.size(); ++i) {
      max_len = std::max<int64_t>(max_len, cu[i] - cu[i - 1]);
    }
    std::vector<int32_t> ws_host(static_cast<size_t>(max_len), 0);
    std::vector<int32_t> we_host(static_cast<size_t>(max_len));
    for (int64_t i = 0; i < max_len; ++i) we_host[static_cast<size_t>(i)] =
        static_cast<int32_t>(i + 1);
    ix_bufs.emplace_back(d, DType::kI32, std::vector<int64_t>{max_len}, ws_host.data());
    Tensor win_start = ix_bufs.back().t();
    ix_bufs.emplace_back(d, DType::kI32, std::vector<int64_t>{max_len}, we_host.data());
    Tensor win_end = ix_bufs.back().t();

    // THE LOGITS BUFFER IS CHUNKED, and the bound is the reason. The indexer's
    // logits are `[query rows, keys]`, so materializing them for a whole
    // request is quadratic in the sequence length — 40 GiB at the released
    // 524288-position context, which is not a buffer anything can allocate.
    // Upstream bounds the same quantity with `get_max_prefill_buffer_size`
    // (deepseek_v2.py:727-729) and chunks its prefill against it. Here the
    // budget is an ELEMENT COUNT and the query chunk is derived from it, so the
    // allocation is flat in the sequence length instead of quadratic.
    constexpr int64_t kIndexerLogitElementBudget = 1 << 24;  // 16 Mi f32 = 64 MiB
    const int64_t chunk =
        std::max<int64_t>(1, std::min<int64_t>(max_len, kIndexerLogitElementBudget / max_len));
    ix_bufs.emplace_back(d, DType::kF32, std::vector<int64_t>{chunk, max_len});
    Tensor logits = ix_bufs.back().t();

    const int64_t soft = static_cast<int64_t>(cu.size()) - 1;
    vt::DsaIndexerLogitsArgs iargs;
    // `self.softmax_scale = self.head_dim**-0.5` (`:709`) and
    // `self.n_head_scale = self.n_head**-0.5` (`:742`).
    iargs.softmax_scale = static_cast<float>(1.0 / std::sqrt(static_cast<double>(ID)));
    iargs.n_head_scale = static_cast<float>(1.0 / std::sqrt(static_cast<double>(IH)));
    // `q_scale` stays null: both dots3-note arms are UNQUANTIZED, where
    // upstream's per-token-group fp8 scale (`:831-838`) is exactly 1. The fp8
    // indexer is owed to the row.
    for (int64_t r = 0; r < soft; ++r) {
      const int64_t o = cu[static_cast<size_t>(r)];
      const int64_t len = cu[static_cast<size_t>(r + 1)] - o;
      // The KEY space is this REQUEST's own tokens, so the selected indices come
      // out as POSITIONS IN THIS REQUEST'S SEQUENCE — which is exactly what
      // `MlaDecodeAttentionArgs::topk_indices` consumes, with no rebasing pass.
      // Upstream needs one (`triton_convert_req_index_to_global_index`,
      // attention.py:760-767) because it hands FlashAttention a flat cache view.
      Tensor k_r = View2(ik, o * ID, len, ID, ID);
      for (int64_t c0 = 0; c0 < len; c0 += chunk) {
        const int64_t c1 = std::min<int64_t>(len, c0 + chunk);
        const int64_t rows = c1 - c0;
        Tensor q_r = View3(iq, (o + c0) * IH * ID, rows, IH, ID, IH * ID, ID, 1);
        Tensor w_r = View2(iw, (o + c0) * IH, rows, IH, IH);
        Tensor lg_r = View2(logits, 0, rows, len, max_len);
        Tensor ws_r = win_start.Slice(0, c0, c1);
        Tensor we_r = win_end.Slice(0, c0, c1);
        Tensor idx_r = View2(topk_idx, (o + c0) * K, rows, K, K);
        Tensor cnt_r = topk_cnt.Slice(0, o + c0, o + c1);
        vt::DsaIndexerLogits(d.q, lg_r, q_r, k_r, w_r, ws_r, we_r, iargs);
        vt::DsaTopkSelect(d.q, idx_r, cnt_r, lg_r, ws_r, we_r);
      }
    }
  }

  // ─── 4. the MLA cache write (W3), BEFORE attention ───────────────────────
  // Upstream order: `do_kv_cache_update(kv_c_normed, k_pe, ...)` at
  // mla_attention.py:592-601, THEN forward_impl at `:602-609`. The prefill path
  // reads the cache for previously-cached CONTEXT only; this step's own K/V come
  // from kv_c_normed / k_pe directly, so writing first is correct.
  Tensor kv_c_write = kv_c_normed.t();
  Tensor k_pe_write = k_pe.t();
  vt::ConcatAndCacheMla(d.q, kv_c_write, k_pe_write, kv_cache, slot_mapping);

  // The attention output in per-head space, [T, N, V] — upstream's
  // `output.view(-1, num_heads, v_head_dim)`.
  DBuf attn(d, dt, {T, N, V});

  // ─── 5a. PREFILL — the materialized-MHA form (mla_attention.py:722-737) ──
  // Runs on the TAIL `q[num_mqa_tokens:]`, because decode tokens are packed
  // first.
  if (prefill_toks > 0) {
    RequireWeight(w.kv_b_proj, "kv_b_proj");
    if (meta.prefill_cu_seqlens_q.data == nullptr) {
      throw std::invalid_argument(
          "MLA block: prefill tokens present but prefill_cu_seqlens_q is unset");
    }
    Tensor q_prefill = View3(q_raw.t(), decode_toks * N * Dqk, prefill_toks, N, Dqk,
                             N * Dqk, Dqk, 1);
    // `kv_b_proj(kv_c_normed)` for THIS step's new tokens (:2371-2373).
    DBuf kv_nope(d, dt, {prefill_toks, N * (P + V)});
    Tensor kv_nope_t = kv_nope.t();
    Tensor kv_c_prefill = View2(kv_c_normed.t(), decode_toks * L, prefill_toks, L, L);
    vt::MatmulBT(d.q, kv_nope_t, kv_c_prefill, w.kv_b_proj);
    Tensor k_nope = View3(kv_nope.t(), 0, prefill_toks, N, P, N * (P + V), P + V, 1);
    Tensor value = View3(kv_nope.t(), P, prefill_toks, N, V, N * (P + V), P + V, 1);
    // At R == 0 the offset is taken as 0: the buffer is zero-width, the concat
    // below copies nothing from it, and `decode_toks * R` would otherwise be a
    // pointer past a 1-byte placeholder allocation.
    Tensor k_pe_prefill =
        View3(k_pe.t(), R > 0 ? decode_toks * R : 0, prefill_toks, 1, R, R, R, 1);
    // `_concat_k_nope_k_pe` (:2374, :2063-2092) — k_pe broadcast over N heads.
    DBuf key(d, dt, {prefill_toks, N, Dqk});
    Tensor key_t = key.t();
    vt::ConcatMlaNopeRope(d.q, key_t, k_nope, k_pe_prefill);

    Tensor prefill_out = View3(attn.t(), decode_toks * N * V, prefill_toks, N, V, N * V, V, 1);
    // The suffix (new-tokens) result + its LSE, plus the chunked-context
    // ping-pong buffers. Only allocated when there IS context to merge.
    const bool has_context = !meta.chunks.empty();
    MlaPrefillContextBuffers bufs{};
    std::vector<DBuf> ctx_bufs;
    DBuf suffix_out(d, dt, {has_context ? prefill_toks : 1, N, V});
    DBuf suffix_lse(d, DType::kF32, {N, has_context ? prefill_toks : 1});
    if (has_context) {
      if (meta.chunk_workspace_tokens <= 0) {
        throw std::invalid_argument(
            "MLA block: chunked context requested but chunk_workspace_tokens is 0 "
            "(size it with DetermineChunkedPrefillWorkspaceSize)");
      }
      ctx_bufs.emplace_back(d, dt,
                            std::vector<int64_t>{meta.chunk_workspace_tokens, L + R});
      bufs.workspace = ctx_bufs.back().t();
      ctx_bufs.emplace_back(d, dt, std::vector<int64_t>{prefill_toks, N, V});
      bufs.chunk_output = ctx_bufs.back().t();
      ctx_bufs.emplace_back(d, DType::kF32, std::vector<int64_t>{N, prefill_toks});
      bufs.chunk_lse = ctx_bufs.back().t();
      ctx_bufs.emplace_back(d, dt, std::vector<int64_t>{prefill_toks, N, V});
      bufs.accum_output = ctx_bufs.back().t();
      ctx_bufs.emplace_back(d, DType::kF32, std::vector<int64_t>{N, prefill_toks});
      bufs.accum_lse = ctx_bufs.back().t();
      ctx_bufs.emplace_back(d, dt, std::vector<int64_t>{prefill_toks, N, V});
      bufs.merge_output = ctx_bufs.back().t();
      ctx_bufs.emplace_back(d, DType::kF32, std::vector<int64_t>{N, prefill_toks});
      bufs.merge_lse = ctx_bufs.back().t();
    }
    MlaUpProjectScratch up_scratch;
    MlaUpProjectFn up = MakeMlaUpProjectFn(d, dims, w, up_scratch);
    Tensor suffix_out_t = suffix_out.t(), suffix_lse_t = suffix_lse.t();
    Tensor kv_cache_ro = kv_cache;
    // dots3-note's sliding layers (#699 W4b-2). 0 — every DeepSeek / MiniCPM3 /
    // Kimi-Linear registration — leaves the call byte-identical; > 0 becomes
    // the `(W - 1, 0)` FlashAttention pair upstream's `run_sliding_window`
    // passes (attention.py:300 @ bc2d63e650), and refuses a windowed prefill
    // that also has chunked context BY NAME.
    ForwardMlaPrefillMha(d.q, prefill_out, q_prefill, key_t, value, kv_cache_ro,
                         meta.prefill_block_table, meta.prefill_cu_seqlens_q, meta.chunks,
                         up, dims.scale, meta.max_query_len,
                         meta.prefill_tokens_with_context, bufs, suffix_out_t,
                         suffix_lse_t, dims.sliding_window);
  }

  // ─── 5b. DECODE — the ABSORBED MQA form (mla_attention.py:739-830) ───────
  if (decode_toks > 0) {
    RequireWeight(w.w_uk_t, "w_uk_t");
    RequireWeight(w.w_uv, "w_uv");
    const int64_t B = decode_toks;
    // `mqa_q_nope = mqa_q[..., :P].transpose(0, 1)` -> (N, B, P) (:743-748).
    Tensor q_nope_t = View3(q_raw.t(), 0, N, B, P, Dqk, N * Dqk, 1);
    // `torch.bmm(mqa_q_nope, self.W_UK_T, out=mqa_ql_nope)` -> (N, B, L) (:789).
    DBuf ql_nope(d, dt, {N, B, L});
    Tensor ql_nope_t = ql_nope.t();
    vt::BatchedMatmul(d.q, ql_nope_t, q_nope_t, w.w_uk_t);
    // `mqa_q = (mqa_ql_nope.transpose(0,1), mqa_q_pe)` (:791-794, :801), which
    // TritonMLAImpl concatenates into one [B, N, 576] query
    // (triton_mla.py:200-201). `concat_mla_q`'s own upstream test covers exactly
    // this non-contiguous transposed nope operand
    // (tests/kernels/test_concat_mla_q.py:37-52).
    DBuf mqa_q(d, dt, {B, N, L + R});
    Tensor mqa_q_t = mqa_q.t();
    Tensor ql_nope_bn = View3(ql_nope.t(), 0, B, N, L, L, B * L, 1);
    // Same at R == 0: `P` is one past the end of the last head's nope slice.
    Tensor q_pe_bn = View3(q_raw.t(), R > 0 ? P : 0, B, N, R, N * Dqk, Dqk, 1);
    vt::ConcatMlaNopeRope(d.q, mqa_q_t, ql_nope_bn, q_pe_bn);
    // `attn_out, lse = self.impl.forward_mqa(mqa_q, kv_cache, ...)` (:812) —
    // still in LATENT space, [B, N, kv_lora_rank].
    DBuf mqa_out(d, dt, {B, N, L});
    Tensor mqa_out_t = mqa_out.t();
    impl.num_heads = static_cast<int>(N);
    impl.head_size = static_cast<int>(dims.head_size());
    impl.scale = dims.scale;
    impl.queue = &d.q;  // W4 deviation (i), wired here.
    // dots3-note's windowed decode (#699 W4b-2). Upstream expresses it as the
    // `Dots3NoteTritonMLAImpl` subclass keeping `self.sliding_window`
    // (attention.py:439-468 @ bc2d63e650); here it is the impl's field, and 0
    // is every DeepSeek / MiniCPM3 / Kimi-Linear caller's value. It is assigned
    // UNCONDITIONALLY rather than under a guard because `impl` is the caller's
    // object and may be reused across layers of DIFFERENT kinds — a guard would
    // let a sliding layer's 513 leak into the next full layer.
    impl.sliding_window = dims.sliding_window;
    v1::AttentionLayer layer{};
    // dots3-note's SPARSE decode (#699 W4b-3c). The selection rides on the
    // decode metadata, exactly as the block table and `seq_lens` do; an EMPTY
    // pair is the absent state and leaves `vt::MlaDecodeAttention` on its
    // byte-identical contiguous key loop. The metadata is COPIED rather than
    // mutated because `meta` is the caller's and is shared by every layer of
    // the step — a sliding layer must not inherit a full layer's selection.
    v1::MLACommonMetadata dec = meta.decode;
    if (run_indexer) {
      dec.topk_indices = topk_idx;
      dec.valid_counts = topk_cnt;
    }
    impl.forward_mqa(layer, mqa_q_t, kv_cache, dec, mqa_out_t, nullptr);
    // `self._v_up_proj(attn_out, out=mqa_output_slice)` (:830, :1024-1034):
    // bmm((N,B,L), W_UV (N,L,V)) written into out.transpose(0,1).
    Tensor x = View3(mqa_out.t(), 0, N, B, L, L, N * L, 1);
    Tensor v_out = View3(attn.t(), 0, N, B, V, V, N * V, 1);
    vt::BatchedMatmul(d.q, v_out, x, w.w_uv);
  }

  // ─── 5c. dots3-note's HEADWISE attention gate (model.py:190-197) ─────────
  // `gate = g_proj(hidden_states)` -> [T, num_heads]; sigmoid in FP32; the
  // whole v_head_dim lane group of head h is scaled by gate[t,h]. Realized as
  // vt::SharedExpertGate over the [T*N, V] view of the attention output, which
  // IS a per-row sigmoid broadcast.
  //
  // THE LOGIT IS BF16, and that is a MEMORY FORMAT decision, not an accident
  // (review finding F2). `g_proj` is built with no `params_dtype`
  // (model.py:292-297), so it inherits the model dtype and upstream's sigmoid
  // input is a BF16 value that `torch.sigmoid(gate.float())` then widens. An
  // f32 GEMM output here would be strictly WIDER than upstream on a model path,
  // which porting.md says a token gate cannot catch — so the GEMM stores bf16
  // and `vt::CastF32` widens it EXACTLY, which is upstream's `.float()`. The
  // f32 copy exists only because vt::SharedExpertGate takes an f32 gate vector.
  //
  // NOTHING is allocated and no op is launched when the weight is empty, which
  // is every DeepSeek registration — the buffers live in `gate_bufs`, which
  // stays empty (review finding F4: a zero-WIDTH DBuf still takes a pool block,
  // because dense_device_glue.h rounds a zero-length request up to 1 byte).
  std::vector<DBuf> gate_bufs;
  gate_bufs.reserve(3);
  Tensor attn_flat = Reshape(attn.t(), {T, N * V});
  if (has_gate) {
    gate_bufs.emplace_back(d, DType::kBF16, std::vector<int64_t>{T, N});
    Tensor gl_bf16 = gate_bufs.back().t();
    vt::MatmulBT(d.q, gl_bf16, hidden, w.attn_gate_proj);
    gate_bufs.emplace_back(d, DType::kF32, std::vector<int64_t>{T, N});
    Tensor gl_f32 = gate_bufs.back().t();
    vt::CastF32(d.q, gl_f32, gl_bf16);  // upstream's `.float()`, exact
    gate_bufs.emplace_back(d, dt, std::vector<int64_t>{T * N, V});
    Tensor go = gate_bufs.back().t();
    Tensor gl_flat = Reshape(gl_f32, {T * N});
    Tensor sd = Reshape(attn.t(), {T * N, V});
    vt::SharedExpertGate(d.q, go, sd, gl_flat);
    attn_flat = Reshape(go, {T, N * V});
  }

  // ─── 6. o_proj (deepseek_v2.py:526; mla.py:181) ──────────────────────────
  RequireWeight(w.o_proj, "o_proj");
  vt::MatmulBT(d.q, out, attn_flat, w.o_proj);
}

}  // namespace mla
}  // namespace vllm
