// LTX-2.5 — the DiT forward: BasicAVTransformerBlock and LTXModel.forward.
// Port of Lightricks LTX-2, packages/ltx-core/src/ltx_core/model/transformer/
// transformer.py (the block) and model.py (the model), with the input preparation
// of transformer_args.py.
//
// f32 throughout: the parity dtype of the L2 gate. See ltx2.h's DTYPE note.
//
// FUSION. `PostSelfAttention` below is an add + RMSNorm and `AdaZero` is an
// RMSNorm + affine — both `vt::FusedChain` shapes. They are deliberately explicit
// host loops here for the same reason MiniMax-H3's correctness forward keeps its
// glue explicit (minimax_h3.cpp, "SCOPE OF THIS TU"): this TU exists to be read
// against upstream line by line and gated exactly, and folding it onto fused
// recipes is the device-resident brick, not this one. Nothing here claims a speed
// result.
#include "vllm/model_executor/models/ltx2.h"

#include <cmath>
#include <cstring>
#include <string>
#include <vector>

#include "vt/backend.h"
#include "vt/ops.h"

namespace vllm {
namespace {

// torch.nn.functional.rms_norm with no weight (ops.py:58 `rms_norm(x, eps=eps)`).
void RmsNormRows(const float* in, float* out, int64_t rows, int64_t width, double eps) {
  for (int64_t r = 0; r < rows; ++r) {
    const float* src = in + r * width;
    double sum = 0.0;
    for (int64_t i = 0; i < width; ++i) sum += static_cast<double>(src[i]) * src[i];
    const float inv = static_cast<float>(1.0 / std::sqrt(sum / static_cast<double>(width) + eps));
    float* dst = out + r * width;
    for (int64_t i = 0; i < width; ++i) dst[i] = src[i] * inv;
  }
}

// torch.nn.LayerNorm(elementwise_affine=False) (model.py:231, :261): biased
// variance, eps inside the square root.
void LayerNormRows(const float* in, float* out, int64_t rows, int64_t width, double eps) {
  for (int64_t r = 0; r < rows; ++r) {
    const float* src = in + r * width;
    double mean = 0.0;
    for (int64_t i = 0; i < width; ++i) mean += src[i];
    mean /= static_cast<double>(width);
    double var = 0.0;
    for (int64_t i = 0; i < width; ++i) {
      const double d = static_cast<double>(src[i]) - mean;
      var += d * d;
    }
    var /= static_cast<double>(width);
    const float inv = static_cast<float>(1.0 / std::sqrt(var + eps));
    const float m = static_cast<float>(mean);
    float* dst = out + r * width;
    for (int64_t i = 0; i < width; ++i) dst[i] = (src[i] - m) * inv;
  }
}

// get_ada_values (transformer.py:191-200) for ONE of the unbound slices:
//   value[b, t, :] = table[table_row, :] + modulation[b, t, mod_index, :]
// `num_params` is the row count of the table SLICE upstream passed in, which is
// also how the flat modulation row is reshaped (:198) — reading it as anything
// else silently mixes the self-attention, feed-forward and cross-attention groups.
// The two indices differ for the AV cross GATE, where upstream hands in
// `scale_shift_table[4:]` (one row) against a one-parameter timestep (:214-215):
// table row 4, modulation index 0.
std::vector<float> AdaValue(const vt::Tensor& table, const float* modulation, int64_t batch,
                            int64_t tokens, int64_t dim, int64_t num_params, int64_t table_row,
                            int64_t mod_index) {
  const float* t = table.Ptr<float>() + table_row * dim;
  std::vector<float> out(static_cast<size_t>(batch * tokens * dim));
  for (int64_t r = 0; r < batch * tokens; ++r) {
    const float* m = modulation + r * num_params * dim + mod_index * dim;
    float* dst = out.data() + r * dim;
    for (int64_t c = 0; c < dim; ++c) dst[c] = t[c] + m[c];
  }
  return out;
}

// The common case: the table row and the modulation slot are the same index.
std::vector<float> AdaValue(const vt::Tensor& table, const float* modulation, int64_t batch,
                            int64_t tokens, int64_t dim, int64_t num_params, int64_t index) {
  return AdaValue(table, modulation, batch, tokens, dim, num_params, index, index);
}

// PytorchAdaZeroFunction (ops.py:50-58): rms_norm(x) * (1 + scale) + shift.
std::vector<float> AdaZero(const float* x, const std::vector<float>& scale,
                           const std::vector<float>& shift, int64_t rows, int64_t width,
                           double eps) {
  std::vector<float> out(static_cast<size_t>(rows * width));
  RmsNormRows(x, out.data(), rows, width, eps);
  for (int64_t r = 0; r < rows; ++r) {
    float* dst = out.data() + r * width;
    const float* s = scale.data() + r * width;
    const float* h = shift.data() + r * width;
    for (int64_t c = 0; c < width; ++c) dst[c] = dst[c] * (1.0f + s[c]) + h[c];
  }
  return out;
}

// PytorchPostSAFunction (ops.py:72-82): x + y * gate, then rms_norm of that sum.
void PostSelfAttention(float* x, const float* y, const std::vector<float>& gate, int64_t rows,
                       int64_t width, double eps, std::vector<float>* normed) {
  for (int64_t r = 0; r < rows; ++r) {
    float* dst = x + r * width;
    const float* src = y + r * width;
    const float* g = gate.data() + r * width;
    for (int64_t c = 0; c < width; ++c) dst[c] += src[c] * g[c];
  }
  normed->resize(static_cast<size_t>(rows * width));
  RmsNormRows(x, normed->data(), rows, width, eps);
}

// apply_cross_attention_adaln (transformer.py:420-447). `prompt_table` is the
// STATIC [2, width] per-block table (:441). `prompt_mod` is the prompt-side AdaLN
// MLP's output for this stream, [batch, 2 * width] — shift row then scale row,
// one row per BATCH element because `_prepare_timestep` ran on the modality's
// per-sample `sigma` (transformer_args.py:274-277). It is nullptr exactly when
// upstream's `prompt_timestep is None` (:442), i.e. use_prompt_adaln_single=false
// — which is what makes the resulting K/V cacheable across denoise steps.
//
// ORDER. Upstream sums the table and the timestep row FIRST and only then
// applies `(1 + scale)` (:441-446). Folding the two additions the other way round
// would round differently, so the sum is materialized here as upstream forms it.
std::vector<float> ModulateContext(const float* context, const vt::Tensor& prompt_table,
                                   const float* prompt_mod, int64_t batch,
                                   int64_t context_tokens, int64_t width) {
  const float* table_shift = prompt_table.Ptr<float>();
  const float* table_scale = prompt_table.Ptr<float>() + width;
  std::vector<float> out(static_cast<size_t>(batch * context_tokens * width));
  std::vector<float> shift_kv(static_cast<size_t>(width));
  std::vector<float> scale_kv(static_cast<size_t>(width));
  for (int64_t b = 0; b < batch; ++b) {
    for (int64_t c = 0; c < width; ++c) {
      shift_kv[static_cast<size_t>(c)] = table_shift[c];
      scale_kv[static_cast<size_t>(c)] = table_scale[c];
    }
    if (prompt_mod != nullptr) {
      const float* m = prompt_mod + b * 2 * width;
      for (int64_t c = 0; c < width; ++c) {
        shift_kv[static_cast<size_t>(c)] += m[c];
        scale_kv[static_cast<size_t>(c)] += m[width + c];
      }
    }
    for (int64_t s = 0; s < context_tokens; ++s) {
      const int64_t r = b * context_tokens + s;
      const float* src = context + r * width;
      float* dst = out.data() + r * width;
      for (int64_t c = 0; c < width; ++c) {
        dst[c] = src[c] * (1.0f + scale_kv[static_cast<size_t>(c)]) +
                 shift_kv[static_cast<size_t>(c)];
      }
    }
  }
  return out;
}

// x = x + y * gate, where `gate` is [batch, 1, width] broadcast over the tokens
// (transformer.py:355-364, :386-395 — the AV cross gate has a single row).
void AddGatedBroadcast(float* x, const std::vector<float>& y, const std::vector<float>& gate,
                       int64_t batch, int64_t tokens, int64_t width) {
  for (int64_t b = 0; b < batch; ++b) {
    const float* g = gate.data() + b * width;
    for (int64_t t = 0; t < tokens; ++t) {
      float* dst = x + (b * tokens + t) * width;
      const float* src = y.data() + (b * tokens + t) * width;
      for (int64_t c = 0; c < width; ++c) dst[c] += src[c] * g[c];
    }
  }
}

// One stream's text cross-attention (transformer.py:223-252 + :420-447).
void TextCrossAttention(vt::Device device, const Ltx2DitParams& params,
                        const Ltx2AttentionWeights& attn, const vt::Tensor& sst,
                        const vt::Tensor& prompt_table, const float* modulation,
                        const float* prompt_modulation, const float* x_normed,
                        const float* context, const float* context_bias,
                        int64_t batch, int64_t tokens, int64_t context_tokens, int64_t width,
                        int64_t heads, int64_t dim_head, const Ltx2CrossKv* kv_in,
                        Ltx2CrossKv* kv_out, float* x) {
  const int64_t coefficient = params.adaln_embedding_coefficient();
  VT_CHECK(params.cross_attention_adaln,
           "ltx2: cross_attention_adaln=false is upstream's plain cross-attention path "
           "(transformer.py:252); LTX-2.5 sets it true and phase L2 gates only that arm");
  // slice(6, 9) — shift_q, scale_q, gate. NOT the self-attention slice(0, 3) and
  // NOT the feed-forward slice(3, 6) (transformer.py:240).
  const std::vector<float> shift_q = AdaValue(sst, modulation, batch, tokens, width, coefficient, 6);
  const std::vector<float> scale_q = AdaValue(sst, modulation, batch, tokens, width, coefficient, 7);
  const std::vector<float> gate = AdaValue(sst, modulation, batch, tokens, width, coefficient, 8);

  std::vector<float> attn_input(static_cast<size_t>(batch * tokens * width));
  for (int64_t r = 0; r < batch * tokens; ++r) {
    const float* src = x_normed + r * width;
    float* dst = attn_input.data() + r * width;
    for (int64_t c = 0; c < width; ++c) {
      dst[c] = src[c] * (1.0f + scale_q[static_cast<size_t>(r * width + c)]) +
               shift_q[static_cast<size_t>(r * width + c)];
    }
  }
  // The modulated context is only needed when the K/V are actually recomputed.
  std::vector<float> encoder;
  if (kv_in == nullptr) {
    encoder = ModulateContext(context, prompt_table, prompt_modulation, batch, context_tokens,
                              width);
  }

  Ltx2AttentionArgs a;
  a.batch = batch;
  a.tokens = tokens;
  a.context_tokens = context_tokens;
  a.query_dim = width;
  a.context_dim = width;
  a.heads = heads;
  a.dim_head = dim_head;
  a.norm_eps = params.norm_eps;
  a.rope_type = params.rope_type;
  a.bias = context_bias;
  a.bias_rows = context_bias != nullptr ? 1 : 0;
  a.kv_in = kv_in;
  a.kv_out = kv_out;
  const std::vector<float> out =
      Ltx2Attention(device, attn, attn_input.data(), kv_in != nullptr ? context : encoder.data(), a);

  for (int64_t r = 0; r < batch * tokens; ++r) {
    float* dst = x + r * width;
    const float* src = out.data() + r * width;
    const float* g = gate.data() + r * width;
    for (int64_t c = 0; c < width; ++c) dst[c] += src[c] * g[c];
  }
}

}  // namespace

void Ltx2TransformerBlockForward(vt::Device device, const Ltx2DitParams& params,
                                 const Ltx2BlockWeights& w, const Ltx2BlockArgs& args,
                                 float* video_x, float* audio_x) {
  const int64_t batch = args.batch;
  const int64_t dim = params.inner_dim();
  const int64_t adim = params.audio_inner_dim();
  const int64_t tv = args.video_tokens;
  const int64_t ta = args.audio_tokens;
  const int64_t coefficient = params.adaln_embedding_coefficient();
  const double eps = params.norm_eps;

  // transformer.py:265-269.
  const bool run_vx = args.video_enabled && video_x != nullptr && tv > 0;
  const bool run_ax = args.audio_enabled && audio_x != nullptr && ta > 0;
  const bool run_a2v = run_vx && audio_x != nullptr && ta > 0;
  const bool run_v2a = run_ax && video_x != nullptr && tv > 0;

  std::vector<float> vx_normed, ax_normed;

  if (run_vx) {
    // slice(0, 3) — shift, scale, gate for the self-attention (transformer.py:272-273).
    const std::vector<float> shift =
        AdaValue(w.scale_shift_table, args.video_timestep_modulation, batch, tv, dim, coefficient, 0);
    const std::vector<float> scale =
        AdaValue(w.scale_shift_table, args.video_timestep_modulation, batch, tv, dim, coefficient, 1);
    const std::vector<float> gate =
        AdaValue(w.scale_shift_table, args.video_timestep_modulation, batch, tv, dim, coefficient, 2);
    const std::vector<float> norm_vx = AdaZero(video_x, scale, shift, batch * tv, dim, eps);

    Ltx2AttentionArgs a;
    a.batch = batch;
    a.tokens = tv;
    a.context_tokens = tv;
    a.query_dim = dim;
    a.context_dim = dim;
    a.heads = params.num_attention_heads;
    a.dim_head = params.attention_head_dim;
    a.norm_eps = eps;
    a.rope_type = params.rope_type;
    a.pe = args.video_pe;
    a.bias = args.video_self_bias;
    a.bias_rows = args.video_self_bias_rows;
    const std::vector<float> msa = Ltx2Attention(device, w.attn1, norm_vx.data(), nullptr, a);
    PostSelfAttention(video_x, msa.data(), gate, batch * tv, dim, eps, &vx_normed);

    TextCrossAttention(device, params, w.attn2, w.scale_shift_table, w.prompt_scale_shift_table,
                       args.video_timestep_modulation, args.video_prompt_modulation,
                       vx_normed.data(), args.video_context,
                       args.video_context_bias, batch, tv, args.video_context_tokens, dim,
                       params.num_attention_heads, params.attention_head_dim,
                       args.prompt_kv_filled ? args.video_prompt_kv : nullptr,
                       args.prompt_kv_filled ? nullptr : args.video_prompt_kv, video_x);
  }

  if (run_ax) {
    const std::vector<float> shift = AdaValue(w.audio_scale_shift_table,
                                              args.audio_timestep_modulation, batch, ta, adim,
                                              coefficient, 0);
    const std::vector<float> scale = AdaValue(w.audio_scale_shift_table,
                                              args.audio_timestep_modulation, batch, ta, adim,
                                              coefficient, 1);
    const std::vector<float> gate = AdaValue(w.audio_scale_shift_table,
                                             args.audio_timestep_modulation, batch, ta, adim,
                                             coefficient, 2);
    const std::vector<float> norm_ax = AdaZero(audio_x, scale, shift, batch * ta, adim, eps);

    Ltx2AttentionArgs a;
    a.batch = batch;
    a.tokens = ta;
    a.context_tokens = ta;
    a.query_dim = adim;
    a.context_dim = adim;
    a.heads = params.audio_num_attention_heads;
    a.dim_head = params.audio_attention_head_dim;
    a.norm_eps = eps;
    a.rope_type = params.rope_type;
    a.pe = args.audio_pe;
    a.bias = args.audio_self_bias;
    a.bias_rows = args.audio_self_bias_rows;
    const std::vector<float> msa = Ltx2Attention(device, w.audio_attn1, norm_ax.data(), nullptr, a);
    PostSelfAttention(audio_x, msa.data(), gate, batch * ta, adim, eps, &ax_normed);

    TextCrossAttention(device, params, w.audio_attn2, w.audio_scale_shift_table,
                       w.audio_prompt_scale_shift_table, args.audio_timestep_modulation,
                       args.audio_prompt_modulation,
                       ax_normed.data(), args.audio_context, args.audio_context_bias, batch, ta,
                       args.audio_context_tokens, adim, params.audio_num_attention_heads,
                       params.audio_attention_head_dim,
                       args.prompt_kv_filled ? args.audio_prompt_kv : nullptr,
                       args.prompt_kv_filled ? nullptr : args.audio_prompt_kv, audio_x);
  }

  // Audio <-> video cross attention (transformer.py:329-397). Both directions
  // read the PRE-cross snapshots so the order of the two does not bias the result.
  if (run_a2v || run_v2a) {
    std::vector<float> vx_pre, ax_pre;
    if (video_x != nullptr) vx_pre.assign(video_x, video_x + batch * tv * dim);
    if (audio_x != nullptr) ax_pre.assign(audio_x, audio_x + batch * ta * adim);

    // get_av_ca_ada_values (transformer.py:202-221): SCALE comes first and SHIFT
    // second — the opposite order from the self-attention slice above.
    auto av_scale_shift = [&](const vt::Tensor& table, const float* ss, int64_t tokens,
                              int64_t width, int64_t first, std::vector<float>* scale,
                              std::vector<float>* shift) {
      *scale = AdaValue(table, ss, batch, tokens, width, 4, first);
      *shift = AdaValue(table, ss, batch, tokens, width, 4, first + 1);
    };
    // The gate is row 4 of the same table, driven by the CROSS modality's sigma
    // and carrying a single token row broadcast over the sequence.
    auto av_gate = [&](const vt::Tensor& table, const float* gate_ts, int64_t width) {
      return AdaValue(table, gate_ts, batch, 1, width, /*num_params=*/1, /*table_row=*/4,
                      /*mod_index=*/0);
    };

    if (run_a2v) {
      std::vector<float> scale_v, shift_v, scale_a, shift_a;
      av_scale_shift(w.scale_shift_table_a2v_ca_video, args.video_cross_scale_shift, tv, dim, 0,
                     &scale_v, &shift_v);
      const std::vector<float> gate =
          av_gate(w.scale_shift_table_a2v_ca_video, args.video_cross_gate, dim);
      const std::vector<float> vq = AdaZero(vx_pre.data(), scale_v, shift_v, batch * tv, dim, eps);
      av_scale_shift(w.scale_shift_table_a2v_ca_audio, args.audio_cross_scale_shift, ta, adim, 0,
                     &scale_a, &shift_a);
      const std::vector<float> akv = AdaZero(ax_pre.data(), scale_a, shift_a, batch * ta, adim, eps);

      Ltx2AttentionArgs a;
      a.batch = batch;
      a.tokens = tv;
      a.context_tokens = ta;
      a.query_dim = dim;    // Q from the VIDEO stream
      a.context_dim = adim;  // K/V from the AUDIO stream
      a.heads = params.audio_num_attention_heads;
      a.dim_head = params.audio_attention_head_dim;
      a.norm_eps = eps;
      a.rope_type = params.rope_type;
      a.pe = args.video_cross_pe;
      a.k_pe = args.audio_cross_pe;
      const std::vector<float> out =
          Ltx2Attention(device, w.audio_to_video_attn, vq.data(), akv.data(), a);
      AddGatedBroadcast(video_x, out, gate, batch, tv, dim);
    }

    if (run_v2a) {
      std::vector<float> scale_a, shift_a, scale_v, shift_v;
      av_scale_shift(w.scale_shift_table_a2v_ca_audio, args.audio_cross_scale_shift, ta, adim, 2,
                     &scale_a, &shift_a);
      const std::vector<float> gate =
          av_gate(w.scale_shift_table_a2v_ca_audio, args.audio_cross_gate, adim);
      const std::vector<float> aq = AdaZero(ax_pre.data(), scale_a, shift_a, batch * ta, adim, eps);
      av_scale_shift(w.scale_shift_table_a2v_ca_video, args.video_cross_scale_shift, tv, dim, 2,
                     &scale_v, &shift_v);
      const std::vector<float> vkv = AdaZero(vx_pre.data(), scale_v, shift_v, batch * tv, dim, eps);

      Ltx2AttentionArgs a;
      a.batch = batch;
      a.tokens = ta;
      a.context_tokens = tv;
      a.query_dim = adim;   // Q from the AUDIO stream
      a.context_dim = dim;  // K/V from the VIDEO stream
      a.heads = params.audio_num_attention_heads;
      a.dim_head = params.audio_attention_head_dim;
      a.norm_eps = eps;
      a.rope_type = params.rope_type;
      a.pe = args.audio_cross_pe;
      a.k_pe = args.video_cross_pe;
      const std::vector<float> out =
          Ltx2Attention(device, w.video_to_audio_attn, aq.data(), vkv.data(), a);
      AddGatedBroadcast(audio_x, out, gate, batch, ta, adim);
    }
  }

  // Feed-forward, slice(3, 6) (transformer.py:399-415).
  if (run_vx) {
    const std::vector<float> shift =
        AdaValue(w.scale_shift_table, args.video_timestep_modulation, batch, tv, dim, coefficient, 3);
    const std::vector<float> scale =
        AdaValue(w.scale_shift_table, args.video_timestep_modulation, batch, tv, dim, coefficient, 4);
    const std::vector<float> gate =
        AdaValue(w.scale_shift_table, args.video_timestep_modulation, batch, tv, dim, coefficient, 5);
    const std::vector<float> scaled = AdaZero(video_x, scale, shift, batch * tv, dim, eps);
    const std::vector<float> ff =
        Ltx2FeedForward(device, w.ff, scaled.data(), batch * tv, dim, 4 * dim);
    for (int64_t r = 0; r < batch * tv; ++r) {
      float* dst = video_x + r * dim;
      for (int64_t c = 0; c < dim; ++c) {
        dst[c] += ff[static_cast<size_t>(r * dim + c)] * gate[static_cast<size_t>(r * dim + c)];
      }
    }
  }

  if (run_ax) {
    const std::vector<float> shift = AdaValue(w.audio_scale_shift_table,
                                              args.audio_timestep_modulation, batch, ta, adim,
                                              coefficient, 3);
    const std::vector<float> scale = AdaValue(w.audio_scale_shift_table,
                                              args.audio_timestep_modulation, batch, ta, adim,
                                              coefficient, 4);
    const std::vector<float> gate = AdaValue(w.audio_scale_shift_table,
                                             args.audio_timestep_modulation, batch, ta, adim,
                                             coefficient, 5);
    const std::vector<float> scaled = AdaZero(audio_x, scale, shift, batch * ta, adim, eps);
    const std::vector<float> ff =
        Ltx2FeedForward(device, w.audio_ff, scaled.data(), batch * ta, adim, 4 * adim);
    for (int64_t r = 0; r < batch * ta; ++r) {
      float* dst = audio_x + r * adim;
      for (int64_t c = 0; c < adim; ++c) {
        dst[c] += ff[static_cast<size_t>(r * adim + c)] * gate[static_cast<size_t>(r * adim + c)];
      }
    }
  }
}

// ---------------------------------------------------------------------------
// LTXModel.forward (model.py:492-538)
// ---------------------------------------------------------------------------

namespace {

// One stream's prepared inputs (TransformerArgs, transformer_args.py:46-70).
struct PreparedStream {
  std::vector<float> x;              // [batch, tokens, width]
  std::vector<float> modulation;     // [batch, tokens, coefficient * width]
  std::vector<float> embedded;       // [batch, tokens, width]
  std::vector<float> context_bias;   // [batch, context_tokens] or empty
  std::vector<float> self_bias;      // [batch, rows, tokens] or empty
  int64_t self_bias_rows = 0;
  Ltx2FreqsCis pe;
  Ltx2FreqsCis cross_pe;
  std::vector<float> cross_scale_shift;  // [batch, tokens, 4 * width]
  std::vector<float> cross_gate;         // [batch, 1, width]
  std::vector<float> prompt_modulation;  // [batch, 1, 2 * width], empty when the flag is off
};

// _prepare_timestep (transformer_args.py:173-186) + AdaLayerNormSingle.
void PrepareTimestep(vt::Device device, const Ltx2AdaLayerNormSingleWeights& adaln,
                     const float* timesteps, int64_t count, int64_t width, int64_t multiplier,
                     std::vector<float>* modulation, std::vector<float>* embedded) {
  std::vector<float> scaled(static_cast<size_t>(count));
  for (int64_t i = 0; i < count; ++i) scaled[static_cast<size_t>(i)] = timesteps[i] *
                                                                       static_cast<float>(multiplier);
  Ltx2AdalnOut out = Ltx2AdaLayerNormSingle(device, adaln, scaled.data(), count, width);
  *modulation = std::move(out.modulation);
  *embedded = std::move(out.embedded);
}

PreparedStream PrepareStream(vt::Device device, const Ltx2DitParams& params,
                             const Ltx2LinearWeight& patchify,
                             const Ltx2AdaLayerNormSingleWeights& adaln,
                             const Ltx2AdaLayerNormSingleWeights& cross_scale_shift_adaln,
                             const Ltx2AdaLayerNormSingleWeights& cross_gate_adaln,
                             const Ltx2AdaLayerNormSingleWeights* prompt_adaln,
                             const Ltx2ModalityInput& m, int64_t width, int64_t in_channels,
                             int64_t n_pos_dims, const std::vector<int64_t>& max_pos,
                             int64_t heads, const Ltx2ModalityInput* cross) {
  vt::Queue q{device, nullptr};
  PreparedStream out;
  const int64_t rows = m.batch * m.tokens;

  // transformer_args.py:268 — x = patchify_proj(latent).
  out.x.resize(static_cast<size_t>(rows * width));
  {
    VT_CHECK(patchify.weight.rank == 2 && patchify.weight.shape[1] == in_channels,
             "ltx2: patchify_proj shape does not match in_channels");
    vt::Tensor a = vt::Tensor::Contiguous(const_cast<float*>(m.latent), vt::DType::kF32,
                                          patchify.weight.device, {rows, in_channels});
    vt::Tensor o = vt::Tensor::Contiguous(out.x.data(), vt::DType::kF32, patchify.weight.device,
                                          {rows, width});
    vt::MatmulBT(q, o, a, patchify.weight);
    const float* b = patchify.bias.Ptr<float>();
    for (int64_t r = 0; r < rows; ++r) {
      float* dst = out.x.data() + r * width;
      for (int64_t c = 0; c < width; ++c) dst[c] += b[c];
    }
  }

  PrepareTimestep(device, adaln, m.timesteps, rows, width, params.timestep_scale_multiplier,
                  &out.modulation, &out.embedded);

  // transformer_args.py:274-277 — the PROMPT-side AdaLN runs on this modality's
  // own SIGMA, [batch], not on its per-token `timesteps`. `_prepare_timestep`
  // applies the same timestep_scale_multiplier, and the result views to
  // [batch, 1, 2 * width]: one row per sample, broadcast over the prompt tokens.
  if (prompt_adaln != nullptr) {
    VT_CHECK(m.sigma != nullptr,
             "ltx2: use_prompt_adaln_single=true needs this modality's sigma "
             "(transformer_args.py:274-277); it drives the prompt-side AdaLN MLP whose output is "
             "added to the cross-attention K/V modulation, and a missing sigma would silently "
             "fall back to the static table");
    std::vector<float> unused;
    PrepareTimestep(device, *prompt_adaln, m.sigma, m.batch, width,
                    params.timestep_scale_multiplier, &out.prompt_modulation, &unused);
  }

  if (m.context_mask != nullptr) {
    out.context_bias = Ltx2PrepareContextMask(m.context_mask, m.batch, m.context_tokens);
  }
  if (m.attention_mask != nullptr) {
    VT_CHECK(m.attention_mask_rows == 1 || m.attention_mask_rows == m.tokens,
             "ltx2: attention_mask rows must be 1 (key-only) or the token count");
    out.self_bias = Ltx2PrepareSelfAttentionMask(
        m.attention_mask, m.batch * m.attention_mask_rows * m.tokens);
    out.self_bias_rows = m.attention_mask_rows;
  }

  out.pe = Ltx2PrecomputeFreqsCis(m.positions, m.batch, m.tokens, n_pos_dims, n_pos_dims,
                                  params.use_middle_indices_grid, width, max_pos,
                                  params.positional_embedding_theta, heads, params.rope_type,
                                  params.double_precision_rope);

  if (cross != nullptr) {
    VT_CHECK(cross->sigma != nullptr,
             "ltx2: the cross modality must supply sigma (transformer_args.py:373-379)");
    // transformer_args.py:364-371 — the cross RoPE is built from the TIME axis
    // only, at audio_cross_attention_dim, and ALWAYS with the middle-indices
    // grid regardless of the model's own flag.
    out.cross_pe = Ltx2PrecomputeFreqsCis(m.positions, m.batch, m.tokens, /*n_pos_dims=*/1,
                                          n_pos_dims, true, params.audio_cross_attention_dim,
                                          {params.cross_pe_max_pos()},
                                          params.positional_embedding_theta, heads,
                                          params.rope_type, params.double_precision_rope);
    // _prepare_cross_attention_timestep (transformer_args.py:388-411).
    std::vector<float> unused;
    PrepareTimestep(device, cross_scale_shift_adaln, m.timesteps, rows, width,
                    params.timestep_scale_multiplier, &out.cross_scale_shift, &unused);
    const float factor = static_cast<float>(params.av_ca_timestep_scale_multiplier) /
                         static_cast<float>(params.timestep_scale_multiplier);
    std::vector<float> gate_ts(static_cast<size_t>(m.batch));
    for (int64_t b = 0; b < m.batch; ++b) {
      gate_ts[static_cast<size_t>(b)] =
          cross->sigma[b] * static_cast<float>(params.timestep_scale_multiplier) * factor;
    }
    Ltx2AdalnOut gate = Ltx2AdaLayerNormSingle(device, cross_gate_adaln, gate_ts.data(), m.batch,
                                               width);
    out.cross_gate = std::move(gate.modulation);
  }
  return out;
}

// _process_output (model.py:472-490).
std::vector<float> ProcessOutput(vt::Device device, const vt::Tensor& table,
                                 const Ltx2LinearWeight& proj, const float* x,
                                 const std::vector<float>& embedded, int64_t rows, int64_t width,
                                 int64_t out_channels, double eps) {
  vt::Queue q{device, nullptr};
  std::vector<float> normed(static_cast<size_t>(rows * width));
  LayerNormRows(x, normed.data(), rows, width, eps);
  const float* shift = table.Ptr<float>();
  const float* scale = table.Ptr<float>() + width;
  // model.py:482-488 forms `scale_shift_values = table + embedded` FIRST and only
  // then applies `x * (1 + scale) + shift`; folding the two additions the other
  // way round would round differently.
  for (int64_t r = 0; r < rows; ++r) {
    float* dst = normed.data() + r * width;
    const float* e = embedded.data() + r * width;
    for (int64_t c = 0; c < width; ++c) {
      const float shift_v = shift[c] + e[c];
      const float scale_v = scale[c] + e[c];
      dst[c] = dst[c] * (1.0f + scale_v) + shift_v;
    }
  }
  std::vector<float> out(static_cast<size_t>(rows * out_channels));
  vt::Tensor a = vt::Tensor::Contiguous(normed.data(), vt::DType::kF32, proj.weight.device,
                                        {rows, width});
  vt::Tensor o = vt::Tensor::Contiguous(out.data(), vt::DType::kF32, proj.weight.device,
                                        {rows, out_channels});
  vt::MatmulBT(q, o, a, proj.weight);
  const float* b = proj.bias.Ptr<float>();
  for (int64_t r = 0; r < rows; ++r) {
    float* dst = out.data() + r * out_channels;
    for (int64_t c = 0; c < out_channels; ++c) dst[c] += b[c];
  }
  return out;
}

// FNV-1a over raw bytes — the same 64-bit constants the golden generator's
// parameter stream uses (scripts/gen-ltx2-goldens.py :: fnv1a64), so there is one
// hash in this port and not two. This is an IDENTITY digest, never a similarity
// one: it hashes the bytes, so -0.0f and 0.0f are different prompts and so are
// two values one f32 ulp apart. That polarity is deliberate — a false refusal
// costs one recompute, a false ACCEPT renders the wrong prompt.
uint64_t Fnv1a64Bytes(const void* data, size_t bytes, uint64_t seed) {
  const unsigned char* p = static_cast<const unsigned char*>(data);
  uint64_t h = seed;
  for (size_t i = 0; i < bytes; ++i) {
    h ^= static_cast<uint64_t>(p[i]);
    h *= 0x100000001B3ULL;
  }
  return h;
}

constexpr uint64_t kFnvOffsetBasis = 0xCBF29CE484222325ULL;

// The digest of one stream's prompt: its context tensor, then its prompt mask.
// The COUNT is folded in first so a shorter buffer can never hash to the same
// value as a longer one that starts with the same bytes.
uint64_t PromptDigest(const float* context, int64_t count) {
  uint64_t h = Fnv1a64Bytes(&count, sizeof(count), kFnvOffsetBasis);
  if (context == nullptr || count <= 0) return h;
  return Fnv1a64Bytes(context, static_cast<size_t>(count) * sizeof(float), h);
}

uint64_t MaskDigest(const int32_t* mask, int64_t count) {
  uint64_t h = Fnv1a64Bytes(&count, sizeof(count), kFnvOffsetBasis);
  if (mask == nullptr || count <= 0) return h;
  return Fnv1a64Bytes(mask, static_cast<size_t>(count) * sizeof(int32_t), h);
}

// Refuse by NAME: say which of the prompt's parts moved, and say what to do
// about it. A pipeline that hits this is reusing one cache across two requests,
// and the actionable repair is `Reset()` (or a per-request cache), never a
// silent recompute that would hide the reuse bug.
void CheckPromptIdentity(const Ltx2PromptIdentity& cached, const Ltx2PromptIdentity& call) {
  const char* changed = nullptr;
  if (cached.batch != call.batch) {
    changed = "the batch size";
  } else if (cached.video_context_tokens != call.video_context_tokens) {
    changed = "the video prompt's token count";
  } else if (cached.video_context_dim != call.video_context_dim) {
    changed = "the video prompt's context width";
  } else if (cached.audio_context_tokens != call.audio_context_tokens) {
    changed = "the audio prompt's token count";
  } else if (cached.audio_context_dim != call.audio_context_dim) {
    changed = "the audio prompt's context width";
  } else if (cached.video_context_digest != call.video_context_digest) {
    changed = "the video prompt's CONTENTS (same length, different prompt)";
  } else if (cached.audio_context_digest != call.audio_context_digest) {
    changed = "the audio prompt's CONTENTS (same length, different prompt)";
  } else if (cached.video_mask_digest != call.video_mask_digest) {
    changed = "the video prompt mask";
  } else if (cached.audio_mask_digest != call.audio_mask_digest) {
    changed = "the audio prompt mask";
  }
  VT_CHECK(changed == nullptr,
           std::string("ltx2: this prompt K/V cache was filled for a DIFFERENT prompt — ") +
               (changed != nullptr ? changed : "") +
               " changed. Reusing it would render the CACHED prompt for this request. Call "
               "Ltx2PromptKvCache::Reset() (or use one cache per request) when the prompt "
               "changes; the cache is only reusable across DENOISE STEPS of one prompt.");
}

}  // namespace

Ltx2PromptIdentity Ltx2PromptIdentityOf(const Ltx2DitParams& params,
                                        const Ltx2ModalityInput& video,
                                        const Ltx2ModalityInput& audio) {
  Ltx2PromptIdentity id;
  id.batch = video.batch;
  id.video_context_tokens = video.context_tokens;
  id.audio_context_tokens = audio.context_tokens;
  id.video_context_dim = params.cross_attention_dim;
  id.audio_context_dim = params.audio_cross_attention_dim;
  id.video_context_digest =
      PromptDigest(video.context, video.batch * video.context_tokens * id.video_context_dim);
  id.audio_context_digest =
      PromptDigest(audio.context, audio.batch * audio.context_tokens * id.audio_context_dim);
  id.video_mask_digest = MaskDigest(video.context_mask, video.batch * video.context_tokens);
  id.audio_mask_digest = MaskDigest(audio.context_mask, audio.batch * audio.context_tokens);
  return id;
}

Ltx2DitOutputs Ltx2DitForward(vt::Device device, const Ltx2DitParams& params,
                              const Ltx2DitWeights& weights, const Ltx2ModalityInput* video,
                              const Ltx2ModalityInput* audio, vt::DType compute_dtype,
                              Ltx2PromptKvCache* cache) {
  VT_CHECK(compute_dtype == vt::DType::kF32,
           "ltx2: phase L2 ships only the f32 parity forward; the bf16 / FP8 / NVFP4 stream "
           "dtypes are phase L6 and are refused rather than silently computed in f32");
  // LTX-2.5 is an LTXModelType.AudioVideo checkpoint (model_configurator.py:47),
  // and that is the only weight contract EnumerateLtx2DitTensors describes. The
  // VideoOnly / AudioOnly types (model.py:31-33) build a DIFFERENT parameter set —
  // no audio stream, no av_ca AdaLN embedders — so they are refused by name rather
  // than served by a path no golden covers. Use `enabled` to run one stream of an
  // AV model, which is what the pipeline itself does.
  VT_CHECK(video != nullptr && audio != nullptr,
           "ltx2: phase L2 ships the AudioVideo model type only; LTXModelType.VideoOnly and "
           "LTXModelType.AudioOnly carry a different weight contract and are not ported");
  const int64_t dim = params.inner_dim();
  const int64_t adim = params.audio_inner_dim();
  // transformer_args.py:197 views the projected context to the STREAM width, so
  // the two must agree; a checkpoint where they do not would reinterpret the
  // sequence length instead of the width.
  VT_CHECK(params.cross_attention_dim == dim,
           "ltx2: cross_attention_dim must equal the video stream width");
  if (cache != nullptr) {
    VT_CHECK(!params.use_prompt_adaln_single,
             "ltx2: the prompt K/V cache is only valid when use_prompt_adaln_single is false "
             "(transformer.py:441-443); with the prompt AdaLN MLP enabled the K/V carry a "
             "timestep term and caching them would be wrong");
  }

  // model.py:222-226 / :252-256 — the module exists only when BOTH flags hold, and
  // `prompt_adaln=getattr(self, "prompt_adaln_single", None)` (:313, :333) is how
  // upstream turns its absence into `prompt_timestep is None`.
  const bool prompt_adaln = params.cross_attention_adaln && params.use_prompt_adaln_single;
  const bool have_both = video != nullptr && audio != nullptr;
  PreparedStream vs, as;
  if (video != nullptr) {
    VT_CHECK(video->context_tokens == 0 || video->context != nullptr,
             "ltx2: the video stream needs a context when context_tokens > 0");
    vs = PrepareStream(device, params, weights.patchify_proj, weights.adaln_single,
                       weights.av_ca_video_scale_shift, weights.av_ca_a2v_gate,
                       prompt_adaln ? &weights.prompt_adaln_single : nullptr, *video, dim,
                       params.in_channels, 3, params.positional_embedding_max_pos,
                       params.num_attention_heads, have_both ? audio : nullptr);
  }
  if (audio != nullptr) {
    as = PrepareStream(device, params, weights.audio_patchify_proj, weights.audio_adaln_single,
                       weights.av_ca_audio_scale_shift, weights.av_ca_v2a_gate,
                       prompt_adaln ? &weights.audio_prompt_adaln_single : nullptr, *audio, adim,
                       params.audio_in_channels, 1, params.audio_positional_embedding_max_pos,
                       params.audio_num_attention_heads, have_both ? video : nullptr);
  }

  const bool use_cache = cache != nullptr;
  if (use_cache) {
    // The cached K/V are a function of the PROMPT (and of nothing else on this
    // path — that is what use_prompt_adaln_single=false buys). A filled cache is
    // therefore bound to one prompt, and a call carrying another one is refused
    // by name here, BEFORE any block reads a stale byte.
    const Ltx2PromptIdentity id = Ltx2PromptIdentityOf(params, *video, *audio);
    if (cache->valid) {
      CheckPromptIdentity(cache->prompt, id);
    } else {
      cache->prompt = id;
      cache->video.assign(static_cast<size_t>(params.num_layers), Ltx2CrossKv{});
      cache->audio.assign(static_cast<size_t>(params.num_layers), Ltx2CrossKv{});
    }
  }

  for (int64_t i = 0; i < params.num_layers; ++i) {
    Ltx2BlockArgs a;
    a.batch = video != nullptr ? video->batch : audio->batch;
    a.video_tokens = video != nullptr ? video->tokens : 0;
    a.audio_tokens = audio != nullptr ? audio->tokens : 0;
    a.video_context_tokens = video != nullptr ? video->context_tokens : 0;
    a.audio_context_tokens = audio != nullptr ? audio->context_tokens : 0;
    a.video_enabled = video != nullptr && video->enabled;
    a.audio_enabled = audio != nullptr && audio->enabled;
    a.video_timestep_modulation = vs.modulation.empty() ? nullptr : vs.modulation.data();
    a.audio_timestep_modulation = as.modulation.empty() ? nullptr : as.modulation.data();
    a.video_prompt_modulation =
        vs.prompt_modulation.empty() ? nullptr : vs.prompt_modulation.data();
    a.audio_prompt_modulation =
        as.prompt_modulation.empty() ? nullptr : as.prompt_modulation.data();
    a.video_cross_scale_shift = vs.cross_scale_shift.empty() ? nullptr : vs.cross_scale_shift.data();
    a.video_cross_gate = vs.cross_gate.empty() ? nullptr : vs.cross_gate.data();
    a.audio_cross_scale_shift = as.cross_scale_shift.empty() ? nullptr : as.cross_scale_shift.data();
    a.audio_cross_gate = as.cross_gate.empty() ? nullptr : as.cross_gate.data();
    a.video_context = video != nullptr ? video->context : nullptr;
    a.audio_context = audio != nullptr ? audio->context : nullptr;
    a.video_context_bias = vs.context_bias.empty() ? nullptr : vs.context_bias.data();
    a.audio_context_bias = as.context_bias.empty() ? nullptr : as.context_bias.data();
    a.video_self_bias = vs.self_bias.empty() ? nullptr : vs.self_bias.data();
    a.video_self_bias_rows = vs.self_bias_rows;
    a.audio_self_bias = as.self_bias.empty() ? nullptr : as.self_bias.data();
    a.audio_self_bias_rows = as.self_bias_rows;
    a.video_pe = video != nullptr ? &vs.pe : nullptr;
    a.audio_pe = audio != nullptr ? &as.pe : nullptr;
    a.video_cross_pe = have_both ? &vs.cross_pe : nullptr;
    a.audio_cross_pe = have_both ? &as.cross_pe : nullptr;
    if (use_cache) {
      a.video_prompt_kv = &cache->video[static_cast<size_t>(i)];
      a.audio_prompt_kv = &cache->audio[static_cast<size_t>(i)];
      a.prompt_kv_filled = cache->valid;
    }
    Ltx2TransformerBlockForward(device, params, weights.blocks[static_cast<size_t>(i)], a,
                                video != nullptr ? vs.x.data() : nullptr,
                                audio != nullptr ? as.x.data() : nullptr);
  }
  if (use_cache) cache->valid = true;

  Ltx2DitOutputs out;
  if (video != nullptr) {
    out.video = ProcessOutput(device, weights.scale_shift_table, weights.proj_out, vs.x.data(),
                              vs.embedded, video->batch * video->tokens, dim, params.out_channels,
                              params.norm_eps);
  }
  if (audio != nullptr) {
    out.audio = ProcessOutput(device, weights.audio_scale_shift_table, weights.audio_proj_out,
                              as.x.data(), as.embedded, audio->batch * audio->tokens, adim,
                              params.audio_out_channels, params.norm_eps);
  }
  return out;
}

}  // namespace vllm
