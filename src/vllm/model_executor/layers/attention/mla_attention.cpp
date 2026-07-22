// The MLA attention block + weight absorption — MLA campaign W6.
// Header (include/vllm/model_executor/models/mla_attention.h) carries the full
// `file:line`-on-both-sides port map; this TU implements it.
#include "vllm/model_executor/models/mla_attention.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "vt/dtype.h"

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

// `yarn_find_correction_dim` (rotary_embedding/common.py:34-42).
double YarnFindCorrectionDim(double num_rotations, int64_t dim, double base,
                             int64_t max_position_embeddings) {
  return (static_cast<double>(dim) *
          std::log(static_cast<double>(max_position_embeddings) /
                   (num_rotations * 2.0 * M_PI))) /
         (2.0 * std::log(base));
}

}  // namespace

void MlaBlockDims::Validate() const {
  if (hidden_size <= 0 || num_heads <= 0 || qk_nope_head_dim <= 0 || qk_rope_head_dim <= 0 ||
      v_head_dim <= 0 || kv_lora_rank <= 0) {
    throw std::invalid_argument("MlaBlockDims: every dimension must be > 0");
  }
  if (q_lora_rank < 0) throw std::invalid_argument("MlaBlockDims: q_lora_rank must be >= 0");
  if (qk_rope_head_dim % 2 != 0) {
    throw std::invalid_argument(
        "MlaBlockDims: qk_rope_head_dim must be even (it is the ROTARY dim; "
        "deepseek_v2.py:1059-1064 builds the rope over qk_rope_head_dim only)");
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
  DBuf kv_c(d, dt, {T, L});
  DBuf k_pe(d, dt, {T, R});
  DBuf q_raw(d, dt, {T, N * Dqk});
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
    DBuf q_c(d, dt, {T, ql});
    Tensor w_qa = fused.Slice(0, 0, ql);
    Tensor w_kva_nope = fused.Slice(0, ql, ql + L);
    Tensor w_kva_rope = fused.Slice(0, ql + L, ql + L + R);
    Tensor q_c_t = q_c.t(), kv_c_t = kv_c.t(), k_pe_t = k_pe.t(), q_raw_t = q_raw.t();
    vt::MatmulBT(d.q, q_c_t, hidden, w_qa);
    vt::MatmulBT(d.q, kv_c_t, hidden, w_kva_nope);
    vt::MatmulBT(d.q, k_pe_t, hidden, w_kva_rope);
    // `q_c = self.q_a_layernorm(q_c)` (mla.py:143) — in-place, like upstream.
    Tensor w_qan = w.q_a_layernorm;
    vt::RmsNorm(d.q, q_c_t, q_c_t, w_qan, vt::RmsNormArgs{dims.rms_norm_eps, false});
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
    Tensor kv_c_t = kv_c.t(), k_pe_t = k_pe.t(), q_raw_t = q_raw.t();
    vt::MatmulBT(d.q, kv_c_t, hidden, kva.Slice(0, 0, L));
    vt::MatmulBT(d.q, k_pe_t, hidden, kva.Slice(0, L, L + R));
    // `q = self.q_proj(hidden_states)[0]` (mla.py:152)
    vt::MatmulBT(d.q, q_raw_t, hidden, w.q_proj);
  }

  // ─── 2. kv_a_layernorm over the LATENT ONLY (deepseek_v2.py:516) ──────────
  // The decoupled rope part is deliberately NOT normed — that asymmetry is the
  // whole reason `kv_a_layernorm` is built over `kv_lora_rank` and not over the
  // full 576-wide projection output.
  RequireWeight(w.kv_a_layernorm, "kv_a_layernorm");
  DBuf kv_c_normed(d, dt, {T, L});
  Tensor kv_c_normed_t = kv_c_normed.t();
  Tensor kv_c_in = kv_c.t();
  Tensor w_kvan = w.kv_a_layernorm;
  vt::RmsNorm(d.q, kv_c_normed_t, kv_c_in, w_kvan,
              vt::RmsNormArgs{dims.rms_norm_eps, false});

  // ─── 3. decoupled RoPE (mla.py:155-167) ──────────────────────────────────
  // `q.view(-1, num_heads, qk_head_dim)`, `k_pe.unsqueeze(1)`, then
  // `q[..., qk_nope_head_dim:], k_pe = rotary_emb(positions,
  //                                q[..., qk_nope_head_dim:], k_pe)`.
  // Only the TRAILING qk_rope_head_dim slice of each query head rotates; the
  // rotation is `is_neox_style=False` (deepseek_v2.py:1059-1064), i.e. the
  // adjacent-pair GPT-J form. Both operands are STRIDED views, which is why W6
  // relaxed vt::RopeFromCache to stride-driven q/k.
  if (R > 0) {
    RequireWeight(w.rope_cos_sin_cache, "rope_cos_sin_cache");
    Tensor q_pe = View3(q_raw.t(), P, T, N, R, N * Dqk, Dqk, 1);
    Tensor k_pe3 = View3(k_pe.t(), 0, T, 1, R, R, R, 1);
    vt::RopeArgs rope;
    rope.rotary_dim = static_cast<int>(R);
    rope.is_neox_style = false;
    vt::RopeFromCache(d.q, q_pe, &k_pe3, positions, w.rope_cos_sin_cache, rope);
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
    Tensor k_pe_prefill = View3(k_pe.t(), decode_toks * R, prefill_toks, 1, R, R, R, 1);
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
    ForwardMlaPrefillMha(d.q, prefill_out, q_prefill, key_t, value, kv_cache_ro,
                         meta.prefill_block_table, meta.prefill_cu_seqlens_q, meta.chunks,
                         up, dims.scale, meta.max_query_len,
                         meta.prefill_tokens_with_context, bufs, suffix_out_t,
                         suffix_lse_t);
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
    Tensor q_pe_bn = View3(q_raw.t(), P, B, N, R, N * Dqk, Dqk, 1);
    vt::ConcatMlaNopeRope(d.q, mqa_q_t, ql_nope_bn, q_pe_bn);
    // `attn_out, lse = self.impl.forward_mqa(mqa_q, kv_cache, ...)` (:812) —
    // still in LATENT space, [B, N, kv_lora_rank].
    DBuf mqa_out(d, dt, {B, N, L});
    Tensor mqa_out_t = mqa_out.t();
    impl.num_heads = static_cast<int>(N);
    impl.head_size = static_cast<int>(dims.head_size());
    impl.scale = dims.scale;
    impl.queue = &d.q;  // W4 deviation (i), wired here.
    v1::AttentionLayer layer{};
    impl.forward_mqa(layer, mqa_q_t, kv_cache, meta.decode, mqa_out_t, nullptr);
    // `self._v_up_proj(attn_out, out=mqa_output_slice)` (:830, :1024-1034):
    // bmm((N,B,L), W_UV (N,L,V)) written into out.transpose(0,1).
    Tensor x = View3(mqa_out.t(), 0, N, B, L, L, N * L, 1);
    Tensor v_out = View3(attn.t(), 0, N, B, V, V, N * V, 1);
    vt::BatchedMatmul(d.q, v_out, x, w.w_uv);
  }

  // ─── 6. o_proj (deepseek_v2.py:526; mla.py:181) ──────────────────────────
  RequireWeight(w.o_proj, "o_proj");
  Tensor attn_flat = Reshape(attn.t(), {T, N * V});
  vt::MatmulBT(d.q, out, attn_flat, w.o_proj);
}

}  // namespace mla
}  // namespace vllm
