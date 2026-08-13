// LTX-2.5 EMBEDDINGS CONNECTOR — see
// include/vllm/model_executor/models/ltx2_connector.h for the upstream mapping
// and the four things that fail silently.
//
// Every block routes through the DiT's OWN parts (`Ltx2Attention`,
// `Ltx2FeedForward`, `Ltx2PrecomputeFreqsCis`, `Ltx2ApplyRotaryEmb`), because
// `_BasicTransformerBlock1D` imports exactly those modules upstream
// (embeddings_connector.py:4-11). A second attention implementation here would be
// a parallel path that could drift from the one the DiT is gated on.
#include "vllm/model_executor/models/ltx2_connector.h"

#include <cmath>
#include <cstddef>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "vt/tensor.h"

namespace vllm {
namespace {

[[noreturn]] void Refuse(const std::string& message) { throw std::runtime_error(message); }

void Require(bool condition, const std::string& message) {
  if (!condition) Refuse(message);
}

// torch's `.to(torch.bfloat16)`: round-to-nearest, ties-to-even on the discarded
// low 16 bits. `learnable_registers` is a BFLOAT16 parameter
// (embeddings_connector.py:135-137), so a port that kept the table at f32 would
// carry ~8 mantissa bits upstream does not have — finite, correctly shaped, and
// wrong only at the padded positions.
float RoundToBf16(float value) {
  uint32_t bits;
  std::memcpy(&bits, &value, sizeof(bits));
  if (((bits >> 23) & 0xFFu) == 0xFFu) return value;  // NaN / Inf pass through
  const uint32_t lsb = (bits >> 16) & 1u;
  bits = (bits + 0x7FFFu + lsb) & 0xFFFF0000u;
  float rounded;
  std::memcpy(&rounded, &bits, sizeof(rounded));
  return rounded;
}

// `rms_norm(x)` with no weight (utils.py:7-12), over the last dimension.
void RmsNormRows(std::vector<float>& x, int64_t rows, int64_t width) {
  for (int64_t r = 0; r < rows; ++r) {
    float* row = x.data() + r * width;
    // f64 SUM ACCUMULATOR, the suite-wide reduction escape (L3 precedent,
    // ltx2_text_encoder.cpp:259-269): upstream's `rms_norm` reduces in f32 but in
    // a BLOCKED order no straight loop reproduces, so accumulating exactly and
    // rounding once is the closest single-rounding approximation to any order.
    // The reciprocal-sqrt below narrows back to f32 immediately, so nothing
    // downstream of this row carries the wider value.
    double sum = 0.0;
    for (int64_t i = 0; i < width; ++i) sum += static_cast<double>(row[i]) * row[i];
    const float inv = static_cast<float>(
        1.0 / std::sqrt(sum / static_cast<double>(width) + kLtx2ConnectorRmsNormEps));
    for (int64_t i = 0; i < width; ++i) row[i] *= inv;
  }
}

// Bind one `Ltx2LinearWeight` from the bag, keeping the storage alive in `views`.
struct WeightViews {
  std::vector<vt::Tensor> keep;
};

vt::Tensor View(const Ltx2VaeWeights& weights, const std::string& name, int64_t rows,
                int64_t cols) {
  const std::vector<float>& buffer = weights.Get(name);
  const int64_t count = cols > 0 ? rows * cols : rows;
  Require(buffer.size() == static_cast<size_t>(count),
          "ltx2 connector: tensor " + name + " has the wrong element count");
  float* data = const_cast<float*>(buffer.data());
  if (cols > 0) {
    return vt::Tensor::Contiguous(data, vt::DType::kF32, vt::Device{}, {rows, cols});
  }
  return vt::Tensor::Contiguous(data, vt::DType::kF32, vt::Device{}, {rows});
}

}  // namespace

std::vector<Ltx2ConnectorTensorSpec> EnumerateLtx2ConnectorTensors(
    const Ltx2ConnectorConfig& config) {
  // torch yields a module's own bare nn.Parameters BEFORE descending into
  // submodules, so `learnable_registers` comes first even though it is declared
  // after `transformer_1d_blocks`.
  const std::string p = config.prefix;
  const int64_t dim = config.inner_dim();
  std::vector<Ltx2ConnectorTensorSpec> specs;
  if (config.num_learnable_registers > 0) {
    specs.push_back({p + "learnable_registers", {config.num_learnable_registers, dim}});
  }
  for (int64_t layer = 0; layer < config.num_layers; ++layer) {
    const std::string b = p + "transformer_1d_blocks." + std::to_string(layer) + ".";
    // Attention's own `named_parameters()` order (attention.py:478-518): the two
    // RMSNorm gains, then to_q / to_k / to_v, then the optional gate logits, then
    // to_out.0.
    specs.push_back({b + "attn1.q_norm.weight", {dim}});
    specs.push_back({b + "attn1.k_norm.weight", {dim}});
    specs.push_back({b + "attn1.to_q.weight", {dim, dim}});
    specs.push_back({b + "attn1.to_q.bias", {dim}});
    specs.push_back({b + "attn1.to_k.weight", {dim, dim}});
    specs.push_back({b + "attn1.to_k.bias", {dim}});
    specs.push_back({b + "attn1.to_v.weight", {dim, dim}});
    specs.push_back({b + "attn1.to_v.bias", {dim}});
    if (config.apply_gated_attention) {
      specs.push_back({b + "attn1.to_gate_logits.weight", {config.num_attention_heads, dim}});
      specs.push_back({b + "attn1.to_gate_logits.bias", {config.num_attention_heads}});
    }
    specs.push_back({b + "attn1.to_out.0.weight", {dim, dim}});
    specs.push_back({b + "attn1.to_out.0.bias", {dim}});
    // FeedForward: `net.0.proj` -> gelu(tanh) -> `net.2`, inner width 4 * dim
    // (feed_forward.py:6-12).
    specs.push_back({b + "ff.net.0.proj.weight", {4 * dim, dim}});
    if (config.ff_bias) specs.push_back({b + "ff.net.0.proj.bias", {4 * dim}});
    specs.push_back({b + "ff.net.2.weight", {dim, 4 * dim}});
    if (config.ff_bias) specs.push_back({b + "ff.net.2.bias", {dim}});
  }
  return specs;
}

Ltx2ConnectorOutput Ltx2ConnectorReplaceRegisters(const Ltx2ConnectorConfig& config,
                                                  const Ltx2VaeWeights& weights,
                                                  const float* hidden_states,
                                                  const float* additive_attention_mask,
                                                  int64_t batch, int64_t seq) {
  Require(config.num_learnable_registers > 0,
          "ltx2 connector: the register substitution needs num_learnable_registers > 0");
  Require(additive_attention_mask != nullptr,
          "ltx2 connector: the register substitution requires an additive attention mask "
          "(embeddings_connector.py:168-170 dereferences it unconditionally)");
  // :144 — the register table is TILED across the sequence, not indexed, so the
  // sequence must be an exact multiple of it.
  Require(seq % config.num_learnable_registers == 0,
          "ltx2 connector: seq_len " + std::to_string(seq) +
              " must be a multiple of num_learnable_registers " +
              std::to_string(config.num_learnable_registers) +
              " (embeddings_connector.py:144)");

  const int64_t dim = config.inner_dim();
  const std::vector<float>& registers = weights.Get(config.prefix + "learnable_registers");
  Require(registers.size() == static_cast<size_t>(config.num_learnable_registers * dim),
          "ltx2 connector: learnable_registers has the wrong element count");

  Ltx2ConnectorOutput out;
  out.hidden_states.resize(static_cast<size_t>(batch * seq * dim));
  // :152 — the mask is REPLACED by zeros: every position is attendable
  // afterwards, including the ones that were padding.
  out.mask.assign(static_cast<size_t>(batch * seq), 0.0f);

  for (int64_t b = 0; b < batch; ++b) {
    for (int64_t s = 0; s < seq; ++s) {
      // :148 — `additive_attention_mask[:, 0, 0, :] >= 0`, so 0.0 keeps and
      // -finfo.max substitutes.
      const bool keep = additive_attention_mask[b * seq + s] >= 0.0f;
      const int64_t register_row = s % config.num_learnable_registers;
      for (int64_t i = 0; i < dim; ++i) {
        const size_t dst = static_cast<size_t>((b * seq + s) * dim + i);
        if (keep) {
          out.hidden_states[dst] = hidden_states[dst];
        } else {
          // `.to(hidden_states.dtype)` (:146) widens the BFLOAT16 table back to
          // f32; the value it widens is already rounded, which is what this
          // mirrors.
          out.hidden_states[dst] =
              RoundToBf16(registers[static_cast<size_t>(register_row * dim + i)]);
        }
      }
    }
  }
  return out;
}

Ltx2ConnectorOutput Ltx2ConnectorForward(const Ltx2ConnectorConfig& config,
                                         const Ltx2VaeWeights& weights,
                                         const float* hidden_states,
                                         const float* additive_attention_mask, int64_t batch,
                                         int64_t seq) {
  Require(hidden_states != nullptr, "ltx2 connector: `hidden_states` is required");
  const int64_t dim = config.inner_dim();
  const int64_t heads = config.num_attention_heads;

  Ltx2ConnectorOutput state;
  if (config.num_learnable_registers > 0) {
    state = Ltx2ConnectorReplaceRegisters(config, weights, hidden_states,
                                          additive_attention_mask, batch, seq);
  } else {
    state.hidden_states.assign(hidden_states, hidden_states + batch * seq * dim);
    if (additive_attention_mask != nullptr) {
      state.mask.assign(additive_attention_mask, additive_attention_mask + batch * seq);
    } else {
      state.mask.assign(static_cast<size_t>(batch * seq), 0.0f);
    }
  }

  // :172-184 — a 1-D position grid over the token index. `use_middle_indices_grid`
  // is FALSE here: `indices_grid` carries one value per token, not a [start, end)
  // pair, unlike the DiT's patch bounds.
  std::vector<double> positions(static_cast<size_t>(batch * seq));
  for (int64_t b = 0; b < batch; ++b) {
    for (int64_t s = 0; s < seq; ++s) {
      positions[static_cast<size_t>(b * seq + s)] = static_cast<double>(s);
    }
  }
  const Ltx2FreqsCis pe = Ltx2PrecomputeFreqsCis(
      positions.data(), batch, seq, /*n_pos_dims=*/1, /*source_n_pos_dims=*/1,
      /*use_middle_indices_grid=*/false, dim, config.positional_embedding_max_pos,
      config.positional_embedding_theta, heads, config.rope_type,
      config.double_precision_rope);

  for (int64_t layer = 0; layer < config.num_layers; ++layer) {
    const std::string b = config.prefix + "transformer_1d_blocks." + std::to_string(layer) + ".";

    Ltx2AttentionWeights attn;
    attn.to_q.weight = View(weights, b + "attn1.to_q.weight", dim, dim);
    attn.to_q.bias = View(weights, b + "attn1.to_q.bias", dim, 0);
    attn.to_k.weight = View(weights, b + "attn1.to_k.weight", dim, dim);
    attn.to_k.bias = View(weights, b + "attn1.to_k.bias", dim, 0);
    attn.to_v.weight = View(weights, b + "attn1.to_v.weight", dim, dim);
    attn.to_v.bias = View(weights, b + "attn1.to_v.bias", dim, 0);
    attn.q_norm = View(weights, b + "attn1.q_norm.weight", dim, 0);
    attn.k_norm = View(weights, b + "attn1.k_norm.weight", dim, 0);
    if (config.apply_gated_attention) {
      attn.to_gate_logits.weight = View(weights, b + "attn1.to_gate_logits.weight", heads, dim);
      attn.to_gate_logits.bias = View(weights, b + "attn1.to_gate_logits.bias", heads, 0);
    }
    attn.to_out.weight = View(weights, b + "attn1.to_out.0.weight", dim, dim);
    attn.to_out.bias = View(weights, b + "attn1.to_out.0.bias", dim, 0);

    Ltx2FeedForwardWeights ff;
    ff.proj_in.weight = View(weights, b + "ff.net.0.proj.weight", 4 * dim, dim);
    ff.proj_out.weight = View(weights, b + "ff.net.2.weight", dim, 4 * dim);
    if (config.ff_bias) {
      ff.proj_in.bias = View(weights, b + "ff.net.0.proj.bias", 4 * dim, 0);
      ff.proj_out.bias = View(weights, b + "ff.net.2.bias", dim, 0);
    }

    // :50 — rms_norm BEFORE the attention, on the residual stream.
    std::vector<float> normed = state.hidden_states;
    RmsNormRows(normed, batch * seq, dim);

    Ltx2AttentionArgs args;
    args.batch = batch;
    args.tokens = seq;
    args.context_tokens = seq;
    args.query_dim = dim;
    args.context_dim = dim;
    args.heads = heads;
    args.dim_head = config.attention_head_dim;
    args.norm_eps = kLtx2ConnectorRmsNormEps;
    args.rope_type = config.rope_type;
    args.pe = &pe;
    // The mask is the additive [batch, 1, seq] broadcast row per batch element —
    // the same form `_prepare_attention_mask` produces. After a register
    // substitution it is all zeros, but it is still PASSED: with registers
    // disabled it is the caller's real mask and dropping it would attend over
    // padding.
    args.bias = state.mask.data();
    args.bias_rows = 1;

    const std::vector<float> attn_out =
        Ltx2Attention(vt::Device{}, attn, normed.data(), nullptr, args);
    for (size_t i = 0; i < state.hidden_states.size(); ++i) {
      state.hidden_states[i] += attn_out[i];
    }

    // :62-67 — a second rms_norm, then the feed-forward, then the residual.
    std::vector<float> ff_in = state.hidden_states;
    RmsNormRows(ff_in, batch * seq, dim);
    const std::vector<float> ff_out =
        Ltx2FeedForward(vt::Device{}, ff, ff_in.data(), batch * seq, dim, 4 * dim);
    for (size_t i = 0; i < state.hidden_states.size(); ++i) {
      state.hidden_states[i] += ff_out[i];
    }
  }

  // :189 — a FINAL rms_norm on top of the two each block already applies.
  RmsNormRows(state.hidden_states, batch * seq, dim);
  return state;
}

Ltx2ConnectorEmbeddings Ltx2ConnectorCreateEmbeddings(
    const Ltx2ConnectorConfig& video_config, const Ltx2VaeWeights& video_weights,
    const float* video_features, const Ltx2ConnectorConfig& audio_config,
    const Ltx2VaeWeights& audio_weights, const float* audio_features,
    const float* additive_attention_mask, int64_t batch, int64_t seq) {
  Require(video_features != nullptr && audio_features != nullptr,
          "ltx2 connector: both modality feature streams are required "
          "(embeddings_processor.py:76-79 refuses one without the other)");
  Require(additive_attention_mask != nullptr,
          "ltx2 connector: the processor requires the additive attention mask; it is what "
          "decides which positions become learnable registers "
          "(embeddings_processor.py:82)");
  const int64_t vdim = video_config.inner_dim();
  const int64_t adim = audio_config.inner_dim();

  // `_compute_right_pad_order` (:23-38): a STABLE descending argsort of the
  // binary mask, so valid tokens move to the front keeping their relative order
  // and padded ones follow. Written as a stable partition because that is what a
  // stable argsort of a 0/1 key IS, and it needs no comparator.
  std::vector<int64_t> order(static_cast<size_t>(batch * seq));
  std::vector<float> reordered_mask(static_cast<size_t>(batch * seq));
  for (int64_t b = 0; b < batch; ++b) {
    int64_t write = 0;
    for (int64_t s = 0; s < seq; ++s) {
      if (additive_attention_mask[b * seq + s] >= 0.0f) order[static_cast<size_t>(b * seq + write++)] = s;
    }
    const int64_t valid = write;
    for (int64_t s = 0; s < seq; ++s) {
      if (additive_attention_mask[b * seq + s] < 0.0f) order[static_cast<size_t>(b * seq + write++)] = s;
    }
    // `new_additive = (new_binary - 1) * finfo.max` (:37): 0 for the valid
    // prefix, -finfo.max for the padded tail.
    for (int64_t s = 0; s < seq; ++s) {
      reordered_mask[static_cast<size_t>(b * seq + s)] =
          s < valid ? 0.0f : -std::numeric_limits<float>::max();
    }
  }

  // `_apply_right_pad_order` (:41-43): gather the feature rows into that order.
  auto gather = [&](const float* src, int64_t width) {
    std::vector<float> out(static_cast<size_t>(batch * seq * width));
    for (int64_t b = 0; b < batch; ++b) {
      for (int64_t s = 0; s < seq; ++s) {
        const int64_t from = order[static_cast<size_t>(b * seq + s)];
        std::memcpy(out.data() + static_cast<size_t>((b * seq + s) * width),
                    src + static_cast<size_t>((b * seq + from) * width),
                    static_cast<size_t>(width) * sizeof(float));
      }
    }
    return out;
  };
  const std::vector<float> video_sorted = gather(video_features, vdim);
  const std::vector<float> audio_sorted = gather(audio_features, adim);

  const Ltx2ConnectorOutput video_out = Ltx2ConnectorForward(
      video_config, video_weights, video_sorted.data(), reordered_mask.data(), batch, seq);
  const Ltx2ConnectorOutput audio_out = Ltx2ConnectorForward(
      audio_config, audio_weights, audio_sorted.data(), reordered_mask.data(), batch, seq);

  Ltx2ConnectorEmbeddings out;
  out.mask.resize(static_cast<size_t>(batch * seq));
  // `_to_binary_mask` (:46-48): `encoded_mask < 0.000001`.
  //
  // THE COMPARISON DIRECTION IS SURPRISING AND IT IS MIRRORED EXACTLY. An
  // additive mask holds 0.0 for a kept position and -finfo(f32).max for a padded
  // one, and BOTH are `< 0.000001` — so this returns 1 at every position, for
  // every mask upstream can produce. The reading a port would arrive at by
  // reasoning about intent (`>= 0`, "keep the unmasked ones") is the OPPOSITE at
  // padded positions, and it is not what either reference does.
  //
  // Checked against both, because a line this surprising is exactly where one
  // implementation being wrong would show: `diffusers`
  // `LTX2TextConnectors.forward` writes `(video_attn_mask < 1e-6).to(int64)` and
  // then the same video-only multiply. They agree, down to the constant.
  //
  // The consequence is that the multiply below is a NO-OP for every mask value
  // that can reach it, and the mask this returns is all ones. That is not a
  // reason to drop either: with registers enabled the connector zeroes the mask
  // itself, so the shipped configuration reaches this line with all-zeros and
  // the identity is CORRECT rather than incidental — and removing upstream's
  // line would silently change a future connector whose output mask is not one
  // of those two values.
  for (int64_t i = 0; i < batch * seq; ++i) {
    out.mask[static_cast<size_t>(i)] = video_out.mask[static_cast<size_t>(i)] < 0.000001f ? 1.0f : 0.0f;
  }
  // :86-87 — the VIDEO encoding is multiplied by that binary mask. The AUDIO
  // encoding is not (:91-93). Mirrored, not tidied.
  out.video = video_out.hidden_states;
  for (int64_t b = 0; b < batch; ++b) {
    for (int64_t s = 0; s < seq; ++s) {
      const float m = out.mask[static_cast<size_t>(b * seq + s)];
      if (m != 0.0f) continue;
      for (int64_t i = 0; i < vdim; ++i) {
        out.video[static_cast<size_t>((b * seq + s) * vdim + i)] = 0.0f;
      }
    }
  }
  out.audio = audio_out.hidden_states;
  return out;
}

}  // namespace vllm
