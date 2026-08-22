// Definitions for MiniMax-Music3's acoustic half. See
// minimax_music3_acoustic.h for the phase, the gates, and why fp32 here is
// upstream's choice rather than a widening.
//
// ─── FLOAT32 WHERE UPSTREAM IS FLOAT32, DOUBLE WHERE IT IS A REDUCTION ──────
//
// Two different things happen below and conflating them is a real defect.
//
// A LONG REDUCTION — a GEMM row, a convolution tap sum, a softmax denominator —
// accumulates in `double` and stores `float`. That is the tree's established
// host-reference convention (`vocoder1d::Conv1d`, `music3::LinearNoBias`) and it
// is wider than torch's blocked float32 sgemm, which is why the full-scale gate
// carries a calibrated tolerance instead of a bit-exact claim.
//
// A SHORT, ELEMENTWISE EXPRESSION — the sigma shift, the Euler step, the CFG
// mix, the overlap blend — is computed in `float`, deliberately, because
// upstream computes it in float32 and the result is BIT-EXACT there. Widening
// those to double is not "more accurate": it produces a different number and
// costs the bit-exact gate. `shift * s / (1 + (shift - 1) * s)` at shift 3 is
// 0.100000024 in float32 and 0.100000001 in double, and the goldens say the
// former.
#include "vllm/model_executor/models/minimax_music3_acoustic.h"
#include "vllm/model_executor/models/vocoder1d.h"
#include "music3_profile.h"  // profile::Timer -- the vocoder.decode_window split (#1664)

#include <algorithm>
#include <cmath>
#include <limits>
#include <numbers>
#include <stdexcept>
#include <string>
#include <utility>

namespace vllm {
namespace models {
namespace music3 {

namespace {

[[noreturn]] void Fail(const std::string& message) { throw std::runtime_error(message); }

void RequireSize(const std::vector<float>& values, int64_t expected, const char* what) {
  if (static_cast<int64_t>(values.size()) != expected) {
    Fail(std::string("MiniMax-Music3 acoustic: ") + what + " is " +
         std::to_string(values.size()) + " values, expected " + std::to_string(expected));
  }
}

// `y = x @ W^T (+ bias)` for row-major x [rows, in_dim] and W [out_dim, in_dim] —
// the torch `nn.Linear` layout. Accumulates in double, stores float32.
std::vector<float> Linear(const std::vector<float>& x, int64_t rows, int64_t in_dim,
                          const std::vector<float>& weight, const std::vector<float>* bias,
                          int64_t out_dim, const char* what) {
  RequireSize(x, rows * in_dim, what);
  RequireSize(weight, out_dim * in_dim, what);
  if (bias != nullptr) RequireSize(*bias, out_dim, what);
  std::vector<float> out(static_cast<size_t>(rows * out_dim));
  for (int64_t r = 0; r < rows; ++r) {
    const float* xr = x.data() + r * in_dim;
    for (int64_t o = 0; o < out_dim; ++o) {
      const float* wo = weight.data() + o * in_dim;
      double acc = bias != nullptr ? (*bias)[static_cast<size_t>(o)] : 0.0;
      for (int64_t i = 0; i < in_dim; ++i) acc += static_cast<double>(xr[i]) * wo[i];
      out[static_cast<size_t>(r * out_dim + o)] = static_cast<float>(acc);
    }
  }
  return out;
}

// `nn.LayerNorm(dim)` over the last axis of [rows, dim]: the BIASED variance,
// then the affine weight AND bias. The AR half's norms are RMSNorms with no
// mean subtraction and no bias; reading one as the other is finite and wrong.
std::vector<float> LayerNorm(const std::vector<float>& x, int64_t rows, int64_t dim,
                             const std::vector<float>& weight, const std::vector<float>& bias) {
  RequireSize(x, rows * dim, "layernorm input");
  RequireSize(weight, dim, "layernorm weight");
  RequireSize(bias, dim, "layernorm bias");
  std::vector<float> out(static_cast<size_t>(rows * dim));
  for (int64_t r = 0; r < rows; ++r) {
    const float* row = x.data() + r * dim;
    double mean = 0.0;
    for (int64_t i = 0; i < dim; ++i) mean += row[i];
    mean /= static_cast<double>(dim);
    double variance = 0.0;
    for (int64_t i = 0; i < dim; ++i) {
      const double centered = static_cast<double>(row[i]) - mean;
      variance += centered * centered;
    }
    variance /= static_cast<double>(dim);
    const double inv = 1.0 / std::sqrt(variance + kDitLayerNormEps);
    for (int64_t i = 0; i < dim; ++i) {
      const double normed = (static_cast<double>(row[i]) - mean) * inv;
      out[static_cast<size_t>(r * dim + i)] =
          static_cast<float>(normed * weight[static_cast<size_t>(i)] +
                             bias[static_cast<size_t>(i)]);
    }
  }
  return out;
}

double Silu(double x) { return x / (1.0 + std::exp(-x)); }

// Non-causal scaled-dot-product attention over [seq, heads * head_dim] with the
// head as the SLOW axis inside a row (`view(batch, seq, heads, head_dim)`).
// `dispatch_attention_fn` is called with no mask (transformer_minimax_music3.py
// :97-103), so every token attends to every token including the prepended
// timestep one — this DiT is NOT causal, unlike the RVQ depth decoder.
std::vector<float> Attention(const std::vector<float>& q, const std::vector<float>& k,
                             const std::vector<float>& v, int64_t seq, int64_t heads,
                             int64_t head_dim) {
  const double inv_sqrt = 1.0 / std::sqrt(static_cast<double>(head_dim));
  std::vector<float> out(static_cast<size_t>(seq * heads * head_dim), 0.0f);
  std::vector<double> scores(static_cast<size_t>(seq));
  for (int64_t h = 0; h < heads; ++h) {
    for (int64_t i = 0; i < seq; ++i) {
      double max_score = -std::numeric_limits<double>::infinity();
      for (int64_t j = 0; j < seq; ++j) {
        double acc = 0.0;
        for (int64_t d = 0; d < head_dim; ++d) {
          acc += static_cast<double>(q[static_cast<size_t>((i * heads + h) * head_dim + d)]) *
                 static_cast<double>(k[static_cast<size_t>((j * heads + h) * head_dim + d)]);
        }
        scores[static_cast<size_t>(j)] = acc * inv_sqrt;
        max_score = std::max(max_score, scores[static_cast<size_t>(j)]);
      }
      double sum = 0.0;
      for (int64_t j = 0; j < seq; ++j) {
        scores[static_cast<size_t>(j)] = std::exp(scores[static_cast<size_t>(j)] - max_score);
        sum += scores[static_cast<size_t>(j)];
      }
      const double inv_sum = 1.0 / sum;
      for (int64_t d = 0; d < head_dim; ++d) {
        double acc = 0.0;
        for (int64_t j = 0; j < seq; ++j) {
          acc += scores[static_cast<size_t>(j)] *
                 static_cast<double>(v[static_cast<size_t>((j * heads + h) * head_dim + d)]);
        }
        out[static_cast<size_t>((i * heads + h) * head_dim + d)] =
            static_cast<float>(acc * inv_sum);
      }
    }
  }
  return out;
}

// One 1x1 `nn.Conv1d(c, c, 1, bias=False)` over [c, length], through the shared
// primitive. `preprocess_conv` and `postprocess_conv` are the only convolutions
// the DiT has, and both are RESIDUAL at the call site.
std::vector<float> PointwiseConv(const std::vector<float>& in, int64_t channels,
                                 int64_t length, const std::vector<float>& weight) {
  RequireSize(in, channels * length, "pointwise conv input");
  RequireSize(weight, channels * channels, "pointwise conv weight");
  int64_t out_len = 0;
  std::vector<float> out =
      vocoder1d::Conv1d(in, channels, length, weight, nullptr, channels, /*kernel=*/1,
                        /*stride=*/1, /*dilation=*/1, /*groups=*/1, &out_len);
  if (out_len != length) Fail("MiniMax-Music3 acoustic: 1x1 conv changed the length");
  return out;
}

std::vector<float> MoveOut(std::map<std::string, std::vector<float>>& tensors,
                           const std::string& name, int64_t expected) {
  const auto it = tensors.find(name);
  if (it == tensors.end()) {
    Fail("MiniMax-Music3 transformer: tensor '" + name + "' is missing");
  }
  if (static_cast<int64_t>(it->second.size()) != expected) {
    Fail("MiniMax-Music3 transformer: tensor '" + name + "' has " +
         std::to_string(it->second.size()) + " values, the config owes " +
         std::to_string(expected));
  }
  return std::move(it->second);
}

std::vector<float> CopyFrom(const std::map<std::string, std::vector<float>>& tensors,
                            const std::string& name, int64_t expected) {
  const auto it = tensors.find(name);
  if (it == tensors.end()) {
    Fail("MiniMax-Music3 vocoder: tensor '" + name + "' is missing (W1 folds weight norm, so "
         "the name carries no _g / _v suffix)");
  }
  if (static_cast<int64_t>(it->second.size()) != expected) {
    Fail("MiniMax-Music3 vocoder: tensor '" + name + "' has " +
         std::to_string(it->second.size()) + " values, the config owes " +
         std::to_string(expected));
  }
  return it->second;
}

}  // namespace

// ---------------------------------------------------------------------------
// The flow-matching scheduler
// ---------------------------------------------------------------------------

std::vector<double> DenoiseSigmaRamp(int64_t num_inference_steps) {
  if (num_inference_steps <= 0) {
    Fail("MiniMax-Music3 scheduler: num_inference_steps must be positive, got " +
         std::to_string(num_inference_steps));
  }
  // np.linspace(1.0, 1.0 / n, n): `arange(n) * step + start`, with the last
  // element ASSIGNED the stop value rather than accumulated to it.
  const double start = 1.0;
  const double stop = 1.0 / static_cast<double>(num_inference_steps);
  std::vector<double> out(static_cast<size_t>(num_inference_steps));
  if (num_inference_steps == 1) {
    out[0] = start;
    return out;
  }
  const double step = (stop - start) / static_cast<double>(num_inference_steps - 1);
  for (int64_t i = 0; i < num_inference_steps; ++i) {
    out[static_cast<size_t>(i)] = static_cast<double>(i) * step + start;
  }
  out[static_cast<size_t>(num_inference_steps - 1)] = stop;
  return out;
}

FlowMatchSchedule FlowMatchSetTimesteps(const std::vector<double>& sigmas,
                                        const MiniMaxMusic3SchedulerConfig& config) {
  if (sigmas.empty()) Fail("MiniMax-Music3 scheduler: the sigma schedule is empty");
  if (config.use_dynamic_shifting) {
    Fail("MiniMax-Music3 scheduler: use_dynamic_shifting needs a `mu` this pipeline never "
         "computes; the shipped scheduler_config sets it false "
         "(scheduling_flow_match_euler_discrete.py:309-310)");
  }
  const int64_t n = static_cast<int64_t>(sigmas.size());
  // `np.array(sigmas).astype(np.float32)` (:343): everything after this is
  // float32 arithmetic, and the shift3 golden can tell.
  std::vector<float> s(static_cast<size_t>(n));
  for (int64_t i = 0; i < n; ++i) s[static_cast<size_t>(i)] = static_cast<float>(sigmas[i]);

  // 1. the resolution-independent shift (:351).
  const float shift = static_cast<float>(config.shift);
  const float shift_minus_one = static_cast<float>(config.shift - 1.0);
  for (float& value : s) {
    value = (shift * value) / (1.0f + shift_minus_one * value);
  }

  FlowMatchSchedule out;
  out.sigmas.reserve(static_cast<size_t>(n + 1));
  // 2. the inversion, and the terminal sigma it appends (:372-377).
  if (config.invert_sigmas) {
    for (float& value : s) value = 1.0f - value;
  }
  const float train = static_cast<float>(config.num_train_timesteps);
  out.timesteps.resize(static_cast<size_t>(n));
  // 3. the train-timestep scale, applied to the FINAL sigmas (:367, :374).
  for (int64_t i = 0; i < n; ++i) {
    out.timesteps[static_cast<size_t>(i)] = s[static_cast<size_t>(i)] * train;
  }
  out.sigmas = s;
  out.sigmas.push_back(config.invert_sigmas ? 1.0f : 0.0f);
  return out;
}

std::vector<float> FlowMatchStep(const std::vector<float>& sample,
                                 const std::vector<float>& velocity, int64_t step_index,
                                 const FlowMatchSchedule& schedule) {
  if (sample.size() != velocity.size()) {
    Fail("MiniMax-Music3 scheduler: the sample has " + std::to_string(sample.size()) +
         " values and the velocity " + std::to_string(velocity.size()));
  }
  if (step_index < 0 || step_index + 1 >= static_cast<int64_t>(schedule.sigmas.size())) {
    Fail("MiniMax-Music3 scheduler: step index " + std::to_string(step_index) +
         " has no successor sigma in a schedule of " +
         std::to_string(schedule.sigmas.size()) + " (the last entry is the appended terminal)");
  }
  const float sigma = schedule.sigmas[static_cast<size_t>(step_index)];
  const float sigma_next = schedule.sigmas[static_cast<size_t>(step_index + 1)];
  const float dt = sigma_next - sigma;
  std::vector<float> out(sample.size());
  for (size_t i = 0; i < sample.size(); ++i) out[i] = sample[i] + dt * velocity[i];
  return out;
}

// ---------------------------------------------------------------------------
// Classifier-free guidance
// ---------------------------------------------------------------------------

std::vector<float> ClassifierFreeGuidanceMix(const std::vector<float>& conditional,
                                             const std::vector<float>& unconditional,
                                             double guidance_scale) {
  if (conditional.size() != unconditional.size()) {
    Fail("MiniMax-Music3 guidance: the conditional row has " +
         std::to_string(conditional.size()) + " values and the unconditional " +
         std::to_string(unconditional.size()));
  }
  const float scale = static_cast<float>(guidance_scale);
  std::vector<float> out(conditional.size());
  for (size_t i = 0; i < conditional.size(); ++i) {
    const float shift = conditional[i] - unconditional[i];
    out[i] = unconditional[i] + scale * shift;
  }
  return out;
}

// ---------------------------------------------------------------------------
// Window bookkeeping
// ---------------------------------------------------------------------------

std::vector<int64_t> ChunkStarts(int64_t num_frames) {
  if (num_frames <= 0) {
    Fail("MiniMax-Music3 denoise: the autoregressive stage produced " +
         std::to_string(num_frames) + " frames, so there is no window to denoise");
  }
  if (num_frames <= kChunkFrames) return {0};
  std::vector<int64_t> out;
  for (int64_t start = 0; start < num_frames - kChunkHop; start += kChunkHop) {
    out.push_back(start);
  }
  return out;
}

WindowCarrySpan ChunkCarrySpan(int64_t latent_length) {
  WindowCarrySpan span;
  span.start = std::max<int64_t>(0, latent_length - 2 * kOverlapLatentLength);
  span.end = std::max<int64_t>(span.start, latent_length - kOverlapLatentLength);
  return span;
}

void BlendOverlap(std::vector<float>& latents, int64_t channels, int64_t length,
                  const std::vector<float>& noise_prompt,
                  const std::vector<float>& previous_latent, int64_t previous_length,
                  int64_t overlap, double time_value) {
  RequireSize(latents, channels * length, "blend latents");
  if (overlap < 0 || overlap > length || overlap > previous_length) {
    Fail("MiniMax-Music3 denoise: overlap " + std::to_string(overlap) +
         " does not fit a window of " + std::to_string(length) +
         " latent frames carrying " + std::to_string(previous_length));
  }
  if (overlap == 0) return;
  RequireSize(noise_prompt, channels * overlap, "blend noise prompt");
  RequireSize(previous_latent, channels * previous_length, "blend previous latent");
  const float t = static_cast<float>(time_value);
  const float coefficient = 1.0f - static_cast<float>(1.0 - 1e-6) * t;
  for (int64_t c = 0; c < channels; ++c) {
    for (int64_t i = 0; i < overlap; ++i) {
      latents[static_cast<size_t>(c * length + i)] =
          coefficient * noise_prompt[static_cast<size_t>(c * overlap + i)] +
          t * previous_latent[static_cast<size_t>(c * previous_length + i)];
    }
  }
}

WaveformCropSpan VocoderCropSpan(int64_t chunk_index, int64_t num_chunks,
                                 int64_t waveform_length, int64_t hop_length) {
  if (num_chunks <= 0 || chunk_index < 0 || chunk_index >= num_chunks) {
    Fail("MiniMax-Music3 decode: window " + std::to_string(chunk_index) + " of " +
         std::to_string(num_chunks) + " is out of range");
  }
  WaveformCropSpan span;
  span.left = chunk_index == 0 ? 0 : kCropLeftLatent * hop_length;
  const int64_t right = chunk_index == num_chunks - 1 ? 0 : kCropRightLatent * hop_length;
  span.right_exclusive = waveform_length - right;
  if (span.right_exclusive < span.left) {
    Fail("MiniMax-Music3 decode: the crop of window " + std::to_string(chunk_index) +
         " emptied a waveform of " + std::to_string(waveform_length) + " samples");
  }
  return span;
}

// ---------------------------------------------------------------------------
// The DiT
// ---------------------------------------------------------------------------

std::vector<std::string> DitTensorNames(const MiniMaxMusic3TransformerConfig& config) {
  std::vector<std::string> out{
      "time_proj.weight",          "time_embed.linear_1.weight", "time_embed.linear_1.bias",
      "time_embed.linear_2.weight", "time_embed.linear_2.bias",  "preprocess_conv.weight",
      "proj_in.weight",
  };
  for (int64_t layer = 0; layer < config.num_layers; ++layer) {
    const std::string prefix = "transformer_blocks." + std::to_string(layer) + ".";
    for (const char* suffix :
         {"norm1.weight", "norm1.bias", "attn.to_q.weight", "attn.to_k.weight",
          "attn.to_v.weight", "attn.to_out.0.weight", "norm2.weight", "norm2.bias",
          "ff_in.weight", "ff_in.bias", "ff_out.weight", "ff_out.bias"}) {
      out.push_back(prefix + suffix);
    }
  }
  out.push_back("proj_out.weight");
  out.push_back("postprocess_conv.weight");
  return out;
}

DitWeights DitWeightsFromTensors(const MiniMaxMusic3TransformerConfig& config,
                                 std::map<std::string, std::vector<float>>& tensors) {
  const int64_t inner = config.inner_dim();
  const int64_t concat = config.concat_channels();
  const int64_t attn_inner = config.num_attention_heads * config.attention_head_dim;
  const int64_t fourier = config.fourier_embedding_dim;

  DitWeights weights;
  weights.time_proj_weight = MoveOut(tensors, "time_proj.weight", fourier / 2);
  weights.time_embed_linear_1_weight =
      MoveOut(tensors, "time_embed.linear_1.weight", inner * fourier);
  weights.time_embed_linear_1_bias = MoveOut(tensors, "time_embed.linear_1.bias", inner);
  weights.time_embed_linear_2_weight =
      MoveOut(tensors, "time_embed.linear_2.weight", inner * inner);
  weights.time_embed_linear_2_bias = MoveOut(tensors, "time_embed.linear_2.bias", inner);
  weights.preprocess_conv_weight = MoveOut(tensors, "preprocess_conv.weight", concat * concat);
  weights.proj_in_weight = MoveOut(tensors, "proj_in.weight", inner * concat);
  for (int64_t layer = 0; layer < config.num_layers; ++layer) {
    const std::string prefix = "transformer_blocks." + std::to_string(layer) + ".";
    DitLayerWeights entry;
    entry.norm1_weight = MoveOut(tensors, prefix + "norm1.weight", inner);
    entry.norm1_bias = MoveOut(tensors, prefix + "norm1.bias", inner);
    entry.to_q = MoveOut(tensors, prefix + "attn.to_q.weight", attn_inner * inner);
    entry.to_k = MoveOut(tensors, prefix + "attn.to_k.weight", attn_inner * inner);
    entry.to_v = MoveOut(tensors, prefix + "attn.to_v.weight", attn_inner * inner);
    entry.to_out = MoveOut(tensors, prefix + "attn.to_out.0.weight", inner * attn_inner);
    entry.norm2_weight = MoveOut(tensors, prefix + "norm2.weight", inner);
    entry.norm2_bias = MoveOut(tensors, prefix + "norm2.bias", inner);
    entry.ff_in_weight = MoveOut(tensors, prefix + "ff_in.weight", 2 * config.ff_inner_dim * inner);
    entry.ff_in_bias = MoveOut(tensors, prefix + "ff_in.bias", 2 * config.ff_inner_dim);
    entry.ff_out_weight = MoveOut(tensors, prefix + "ff_out.weight", inner * config.ff_inner_dim);
    entry.ff_out_bias = MoveOut(tensors, prefix + "ff_out.bias", inner);
    weights.layers.push_back(std::move(entry));
  }
  weights.proj_out_weight = MoveOut(tensors, "proj_out.weight", config.in_channels * inner);
  weights.postprocess_conv_weight =
      MoveOut(tensors, "postprocess_conv.weight", config.in_channels * config.in_channels);
  return weights;
}

std::vector<float> FourierTimeEmbedding(double timestep, const std::vector<float>& weight,
                                        int64_t embedding_dim) {
  if (embedding_dim <= 0 || embedding_dim % 2 != 0) {
    Fail("MiniMax-Music3 DiT: fourier_embedding_dim must be positive and even, got " +
         std::to_string(embedding_dim));
  }
  RequireSize(weight, embedding_dim / 2, "fourier projection");
  // `2 * pi * t` in float32, then the 1-term matmul against each projection
  // row. COS first, SIN second (transformer_minimax_music3.py:39).
  const float scaled =
      static_cast<float>(2.0 * std::numbers::pi_v<double>) * static_cast<float>(timestep);
  std::vector<float> out(static_cast<size_t>(embedding_dim));
  for (int64_t i = 0; i < embedding_dim / 2; ++i) {
    const double angle = static_cast<double>(scaled * weight[static_cast<size_t>(i)]);
    out[static_cast<size_t>(i)] = static_cast<float>(std::cos(angle));
    out[static_cast<size_t>(embedding_dim / 2 + i)] = static_cast<float>(std::sin(angle));
  }
  return out;
}

std::vector<float> DitTimestepEmbedding(const std::vector<float>& fourier,
                                        const MiniMaxMusic3TransformerConfig& config,
                                        const DitWeights& weights) {
  const int64_t inner = config.inner_dim();
  std::vector<float> hidden =
      Linear(fourier, 1, config.fourier_embedding_dim, weights.time_embed_linear_1_weight,
             &weights.time_embed_linear_1_bias, inner, "time_embed.linear_1");
  for (float& value : hidden) value = static_cast<float>(Silu(value));
  return Linear(hidden, 1, inner, weights.time_embed_linear_2_weight,
                &weights.time_embed_linear_2_bias, inner, "time_embed.linear_2");
}

DitRotaryTables BuildDitRotaryTables(int64_t seq_len, int64_t rotary_dim, double theta) {
  if (seq_len <= 0 || rotary_dim <= 0 || rotary_dim % 2 != 0) {
    Fail("MiniMax-Music3 DiT: rotary tables need a positive sequence and an even rotary_dim, "
         "got seq " + std::to_string(seq_len) + " rotary_dim " + std::to_string(rotary_dim));
  }
  const int64_t half = rotary_dim / 2;
  std::vector<float> inv_freq(static_cast<size_t>(half));
  for (int64_t j = 0; j < half; ++j) {
    const double exponent = static_cast<double>(2 * j) / static_cast<double>(rotary_dim);
    inv_freq[static_cast<size_t>(j)] = static_cast<float>(1.0 / std::pow(theta, exponent));
  }
  DitRotaryTables tables;
  tables.cos.resize(static_cast<size_t>(seq_len * rotary_dim));
  tables.sin.resize(static_cast<size_t>(seq_len * rotary_dim));
  for (int64_t s = 0; s < seq_len; ++s) {
    for (int64_t j = 0; j < half; ++j) {
      const double angle =
          static_cast<double>(static_cast<float>(s) * inv_freq[static_cast<size_t>(j)]);
      const float c = static_cast<float>(std::cos(angle));
      const float sn = static_cast<float>(std::sin(angle));
      // `torch.cat((freqs, freqs), dim=-1)`: the second half REPEATS the first.
      tables.cos[static_cast<size_t>(s * rotary_dim + j)] = c;
      tables.cos[static_cast<size_t>(s * rotary_dim + half + j)] = c;
      tables.sin[static_cast<size_t>(s * rotary_dim + j)] = sn;
      tables.sin[static_cast<size_t>(s * rotary_dim + half + j)] = sn;
    }
  }
  return tables;
}

void ApplyPartialRotary(std::vector<float>& x, int64_t seq_len, int64_t heads,
                        int64_t head_dim, const DitRotaryTables& tables) {
  RequireSize(x, seq_len * heads * head_dim, "rotary input");
  const int64_t rotary_dim =
      seq_len > 0 ? static_cast<int64_t>(tables.cos.size()) / seq_len : 0;
  if (rotary_dim <= 0 || rotary_dim > head_dim ||
      static_cast<int64_t>(tables.cos.size()) != seq_len * rotary_dim ||
      tables.sin.size() != tables.cos.size()) {
    Fail("MiniMax-Music3 DiT: rotary tables of " + std::to_string(tables.cos.size()) +
         " values do not describe a rotary window inside head_dim " +
         std::to_string(head_dim));
  }
  const int64_t half = rotary_dim / 2;
  for (int64_t s = 0; s < seq_len; ++s) {
    const float* cos = tables.cos.data() + s * rotary_dim;
    const float* sin = tables.sin.data() + s * rotary_dim;
    for (int64_t h = 0; h < heads; ++h) {
      float* row = x.data() + (s * heads + h) * head_dim;
      for (int64_t d = 0; d < half; ++d) {
        // rotate_half = cat(-second, first) OVER THE ROTARY WINDOW; everything
        // from `rotary_dim` on is left exactly as it was.
        const float first = row[d];
        const float second = row[half + d];
        row[d] = first * cos[d] + (-second) * sin[d];
        row[half + d] = second * cos[half + d] + first * sin[half + d];
      }
    }
  }
}

std::vector<float> DitForward(const std::vector<float>& latents, int64_t length,
                              const std::vector<float>& condition, double timestep,
                              const MiniMaxMusic3TransformerConfig& config,
                              const DitWeights& weights) {
  if (length <= 0) {
    Fail("MiniMax-Music3 DiT: a window of " + std::to_string(length) +
         " latent frames has nothing to denoise");
  }
  const int64_t in_channels = config.in_channels;
  const int64_t concat = config.concat_channels();
  const int64_t inner = config.inner_dim();
  const int64_t heads = config.num_attention_heads;
  const int64_t head_dim = config.attention_head_dim;
  const int64_t attn_inner = heads * head_dim;
  const int64_t ff = config.ff_inner_dim;
  RequireSize(latents, in_channels * length, "DiT latents [in_channels, length]");
  RequireSize(condition, length * config.condition_dim,
              "DiT condition [length, condition_dim]");
  if (static_cast<int64_t>(weights.layers.size()) != config.num_layers) {
    Fail("MiniMax-Music3 DiT: the weights carry " + std::to_string(weights.layers.size()) +
         " blocks, the config declares " + std::to_string(config.num_layers));
  }

  // `cat((hidden_states, zeros_like(hidden_states), encoder_hidden_states.T))`
  // (:218-219). The middle block is a genuine ZERO PAD, not a second copy.
  std::vector<float> stacked(static_cast<size_t>(concat * length), 0.0f);
  for (int64_t c = 0; c < in_channels; ++c) {
    for (int64_t t = 0; t < length; ++t) {
      stacked[static_cast<size_t>(c * length + t)] =
          latents[static_cast<size_t>(c * length + t)];
    }
  }
  for (int64_t c = 0; c < config.condition_dim; ++c) {
    for (int64_t t = 0; t < length; ++t) {
      stacked[static_cast<size_t>((2 * in_channels + c) * length + t)] =
          condition[static_cast<size_t>(t * config.condition_dim + c)];
    }
  }
  // RESIDUAL 1x1 convolution (:220).
  const std::vector<float> pre =
      PointwiseConv(stacked, concat, length, weights.preprocess_conv_weight);
  std::vector<float> transposed(static_cast<size_t>(length * concat));
  for (int64_t t = 0; t < length; ++t) {
    for (int64_t c = 0; c < concat; ++c) {
      transposed[static_cast<size_t>(t * concat + c)] =
          pre[static_cast<size_t>(c * length + t)] + stacked[static_cast<size_t>(c * length + t)];
    }
  }

  const std::vector<float> temb = DitTimestepEmbedding(
      FourierTimeEmbedding(timestep, weights.time_proj_weight, config.fourier_embedding_dim),
      config, weights);
  const std::vector<float> projected =
      Linear(transposed, length, concat, weights.proj_in_weight, nullptr, inner, "proj_in");

  // The timestep embedding is PREPENDED as one extra token (:227) that the
  // rotary sees and `proj_out` then drops (:236).
  const int64_t seq = length + 1;
  std::vector<float> hidden(static_cast<size_t>(seq * inner));
  for (int64_t i = 0; i < inner; ++i) hidden[static_cast<size_t>(i)] = temb[static_cast<size_t>(i)];
  for (int64_t t = 0; t < length; ++t) {
    for (int64_t i = 0; i < inner; ++i) {
      hidden[static_cast<size_t>((t + 1) * inner + i)] =
          projected[static_cast<size_t>(t * inner + i)];
    }
  }
  const DitRotaryTables tables = BuildDitRotaryTables(seq, config.rotary_dim, kDitRotaryTheta);

  for (const DitLayerWeights& layer : weights.layers) {
    const std::vector<float> normed =
        LayerNorm(hidden, seq, inner, layer.norm1_weight, layer.norm1_bias);
    std::vector<float> q = Linear(normed, seq, inner, layer.to_q, nullptr, attn_inner, "to_q");
    std::vector<float> k = Linear(normed, seq, inner, layer.to_k, nullptr, attn_inner, "to_k");
    const std::vector<float> v =
        Linear(normed, seq, inner, layer.to_v, nullptr, attn_inner, "to_v");
    ApplyPartialRotary(q, seq, heads, head_dim, tables);
    ApplyPartialRotary(k, seq, heads, head_dim, tables);
    const std::vector<float> attended = Attention(q, k, v, seq, heads, head_dim);
    const std::vector<float> projected_attn =
        Linear(attended, seq, attn_inner, layer.to_out, nullptr, inner, "to_out");
    for (size_t i = 0; i < hidden.size(); ++i) hidden[i] += projected_attn[i];

    const std::vector<float> normed2 =
        LayerNorm(hidden, seq, inner, layer.norm2_weight, layer.norm2_bias);
    const std::vector<float> gated = Linear(normed2, seq, inner, layer.ff_in_weight,
                                            &layer.ff_in_bias, 2 * ff, "ff_in");
    // `gate_states, gate = ff_in(...).chunk(2, -1)` then
    // `ff_out(gate_states * silu(gate))` (:142-143). The FIRST half is the
    // value and the SECOND is what SiLU runs on; swapping them is a different,
    // equally finite network.
    std::vector<float> activated(static_cast<size_t>(seq * ff));
    for (int64_t t = 0; t < seq; ++t) {
      for (int64_t i = 0; i < ff; ++i) {
        const double value = gated[static_cast<size_t>(t * 2 * ff + i)];
        const double gate = gated[static_cast<size_t>(t * 2 * ff + ff + i)];
        activated[static_cast<size_t>(t * ff + i)] = static_cast<float>(value * Silu(gate));
      }
    }
    const std::vector<float> ff_out = Linear(activated, seq, ff, layer.ff_out_weight,
                                             &layer.ff_out_bias, inner, "ff_out");
    for (size_t i = 0; i < hidden.size(); ++i) hidden[i] += ff_out[i];
  }

  // Drop the timestep token, project, transpose back to channel-major.
  const std::vector<float> tail(hidden.begin() + static_cast<std::ptrdiff_t>(inner),
                                hidden.end());
  const std::vector<float> out_rows =
      Linear(tail, length, inner, weights.proj_out_weight, nullptr, in_channels, "proj_out");
  std::vector<float> out(static_cast<size_t>(in_channels * length));
  for (int64_t c = 0; c < in_channels; ++c) {
    for (int64_t t = 0; t < length; ++t) {
      out[static_cast<size_t>(c * length + t)] =
          out_rows[static_cast<size_t>(t * in_channels + c)];
    }
  }
  // RESIDUAL 1x1 convolution (:238).
  const std::vector<float> post =
      PointwiseConv(out, in_channels, length, weights.postprocess_conv_weight);
  for (size_t i = 0; i < out.size(); ++i) out[i] += post[i];
  return out;
}

// ---------------------------------------------------------------------------
// The vocoder
// ---------------------------------------------------------------------------

VocoderWeights VocoderWeightsFromLoader(const MiniMaxMusic3VocoderConfig& config,
                                        const MiniMaxMusic3VocoderWeights& loaded) {
  const int64_t hidden = config.decoder_hidden_dim;
  const int64_t input_dim = config.decoder_input_dim;
  VocoderWeights weights;
  weights.dec_in_proj_weight =
      CopyFrom(loaded.tensors, "dec_in_proj.weight", input_dim * config.stream_channels());
  weights.dec_in_proj_bias = CopyFrom(loaded.tensors, "dec_in_proj.bias", input_dim);
  weights.conv_in_weight = CopyFrom(loaded.tensors, "conv_in.weight", hidden * input_dim * 7);
  weights.conv_in_bias = CopyFrom(loaded.tensors, "conv_in.bias", hidden);

  int64_t last_output = hidden;
  for (size_t index = 0; index < config.upsampling_ratios.size(); ++index) {
    const int64_t stride = config.upsampling_ratios[index];
    const int64_t input = hidden >> index;
    const int64_t output = hidden >> (index + 1);
    last_output = output;
    const std::string block = "blocks." + std::to_string(index) + ".";
    VocoderBlockWeights entry;
    entry.snake1_alpha = CopyFrom(loaded.tensors, block + "snake1.alpha", input);
    entry.conv_t1_weight =
        CopyFrom(loaded.tensors, block + "conv_t1.weight", input * output * 2 * stride);
    entry.conv_t1_bias = CopyFrom(loaded.tensors, block + "conv_t1.bias", output);
    for (int unit = 1; unit <= kVocoderResidualUnits; ++unit) {
      const std::string prefix = block + "res_unit" + std::to_string(unit) + ".";
      VocoderResidualUnitWeights res;
      res.snake1_alpha = CopyFrom(loaded.tensors, prefix + "snake1.alpha", output);
      res.conv1_weight = CopyFrom(loaded.tensors, prefix + "conv1.weight", output * output * 7);
      res.conv1_bias = CopyFrom(loaded.tensors, prefix + "conv1.bias", output);
      res.snake2_alpha = CopyFrom(loaded.tensors, prefix + "snake2.alpha", output);
      res.conv2_weight = CopyFrom(loaded.tensors, prefix + "conv2.weight", output * output);
      res.conv2_bias = CopyFrom(loaded.tensors, prefix + "conv2.bias", output);
      entry.res_units.push_back(std::move(res));
    }
    weights.blocks.push_back(std::move(entry));
  }
  weights.snake_out_alpha = CopyFrom(loaded.tensors, "snake_out.alpha", last_output);
  weights.conv_out_weight = CopyFrom(loaded.tensors, "conv_out.weight", last_output * 7);
  weights.conv_out_bias = CopyFrom(loaded.tensors, "conv_out.bias", 1);
  return weights;
}

void VocoderSnake(std::vector<float>& x, int64_t channels, int64_t length,
                  const std::vector<float>& alpha) {
  // A LEAF of the `vocoder.decode_window` split (#1664). The bracket is free
  // with `VLLM_CPP_MUSIC3_PROFILE` unset: one predicted branch, no clock read.
  profile::Timer timer("vocoder.snake");
  RequireSize(x, channels * length, "snake input");
  RequireSize(alpha, channels, "snake alpha");
  // W1's finding: Music3's `MiniMaxMusic3Snake1d` IS this function with a null
  // `beta` and `logscale = false`, down to `kSnakeEps`.
  vocoder1d::SnakeActivation(x, channels, length, alpha, /*beta=*/nullptr,
                             /*logscale=*/false);
}

std::vector<float> VocoderResidualUnit(const std::vector<float>& in, int64_t dim,
                                       int64_t length, int64_t dilation,
                                       const VocoderResidualUnitWeights& weights,
                                       int64_t* out_len) {
  RequireSize(in, dim * length, "residual unit input");
  std::vector<float> hidden;
  {
    profile::Timer timer("vocoder.copy");
    hidden = in;
  }
  VocoderSnake(hidden, dim, length, weights.snake1_alpha);
  // `pad = (7 - 1) * dilation // 2` (minimax_music3_vocoder.py:39) — a
  // symmetric ZERO pad, the `padding=` argument of the nn.Conv1d itself.
  const int64_t pad = (7 - 1) * dilation / 2;
  int64_t padded_len = 0;
  std::vector<float> padded;
  {
    profile::Timer timer("vocoder.pad");
    padded = vocoder1d::Pad1d(hidden, dim, length, pad, pad, /*replicate=*/false, &padded_len);
  }
  int64_t conv_len = 0;
  std::vector<float> conv;
  {
    profile::Timer timer("vocoder.conv1d");
    conv = vocoder1d::Conv1d(padded, dim, padded_len, weights.conv1_weight, &weights.conv1_bias,
                             dim, /*kernel=*/7, /*stride=*/1, dilation, /*groups=*/1, &conv_len);
  }
  if (conv_len != length) {
    Fail("MiniMax-Music3 vocoder: residual unit at dilation " + std::to_string(dilation) +
         " changed the length from " + std::to_string(length) + " to " +
         std::to_string(conv_len));
  }
  VocoderSnake(conv, dim, conv_len, weights.snake2_alpha);
  int64_t final_len = 0;
  std::vector<float> out;
  {
    profile::Timer timer("vocoder.conv1d");
    out = vocoder1d::Conv1d(conv, dim, conv_len, weights.conv2_weight, &weights.conv2_bias, dim,
                            /*kernel=*/1, /*stride=*/1, /*dilation=*/1, /*groups=*/1, &final_len);
  }
  {
    profile::Timer timer("vocoder.residual_add");
    for (size_t i = 0; i < out.size(); ++i) out[i] += in[i];
  }
  *out_len = final_len;
  return out;
}

std::vector<float> VocoderBlock(const std::vector<float>& in, int64_t input_dim,
                                int64_t output_dim, int64_t length, int64_t stride,
                                const VocoderBlockWeights& weights, int64_t* out_len) {
  RequireSize(in, input_dim * length, "vocoder block input");
  if (static_cast<int64_t>(weights.res_units.size()) != kVocoderResidualUnits) {
    Fail("MiniMax-Music3 vocoder: a block carries " +
         std::to_string(weights.res_units.size()) + " residual units, upstream has " +
         std::to_string(kVocoderResidualUnits));
  }
  std::vector<float> hidden;
  {
    profile::Timer timer("vocoder.copy");
    hidden = in;
  }
  VocoderSnake(hidden, input_dim, length, weights.snake1_alpha);
  // `padding=ceil(stride / 2)` (minimax_music3_vocoder.py:57).
  const int64_t padding = (stride + 1) / 2;
  int64_t up_len = 0;
  std::vector<float> up;
  {
    profile::Timer timer("vocoder.conv_transpose");
    up = vocoder1d::ConvTranspose1d(hidden, input_dim, length, weights.conv_t1_weight,
                                    &weights.conv_t1_bias, output_dim, /*kernel=*/2 * stride,
                                    stride, padding, /*groups=*/1, &up_len);
  }
  for (int64_t unit = 0; unit < kVocoderResidualUnits; ++unit) {
    int64_t next = 0;
    up = VocoderResidualUnit(up, output_dim, up_len, kVocoderResidualDilations[unit],
                             weights.res_units[static_cast<size_t>(unit)], &next);
    up_len = next;
  }
  *out_len = up_len;
  return up;
}

std::vector<float> VocoderDecode(const std::vector<float>& latents, int64_t length,
                                 const MiniMaxMusic3VocoderConfig& config,
                                 const VocoderWeights& weights, int64_t* out_samples) {
  if (length <= 0) {
    Fail("MiniMax-Music3 vocoder: a window of " + std::to_string(length) +
         " latent frames decodes to no audio");
  }
  RequireSize(latents, config.latent_channels * length, "vocoder latents");
  if (weights.blocks.size() != config.upsampling_ratios.size()) {
    Fail("MiniMax-Music3 vocoder: the weights carry " + std::to_string(weights.blocks.size()) +
         " blocks, the config declares " + std::to_string(config.upsampling_ratios.size()));
  }
  const int64_t stream = config.stream_channels();
  const int64_t hop = config.hop_length();

  std::vector<float> waveform;
  int64_t samples = 0;
  for (int64_t channel = 0; channel < 2; ++channel) {
    // `latents.reshape(batch * 2, latent_channels // 2, length)` (:110): a
    // CONTIGUOUS split, so the FIRST `stream` channels are the left stream and
    // the second `stream` the right. Interleaving them instead is the plausible
    // reading that produces a correctly shaped, wrong waveform.
    std::vector<float> hidden(static_cast<size_t>(stream * length));
    for (int64_t c = 0; c < stream; ++c) {
      for (int64_t t = 0; t < length; ++t) {
        hidden[static_cast<size_t>(c * length + t)] =
            latents[static_cast<size_t>((channel * stream + c) * length + t)];
      }
    }
    int64_t current = length;
    int64_t produced = 0;
    {
      profile::Timer timer("vocoder.conv1d");
      hidden = vocoder1d::Conv1d(hidden, stream, current, weights.dec_in_proj_weight,
                                 &weights.dec_in_proj_bias, config.decoder_input_dim,
                                 /*kernel=*/1, /*stride=*/1, /*dilation=*/1, /*groups=*/1,
                                 &produced);
    }
    current = produced;
    // `nn.Conv1d(..., kernel_size=7, padding=3)` (:89).
    int64_t padded_len = 0;
    {
      profile::Timer timer("vocoder.pad");
      hidden = vocoder1d::Pad1d(hidden, config.decoder_input_dim, current, 3, 3,
                                /*replicate=*/false, &padded_len);
    }
    {
      profile::Timer timer("vocoder.conv1d");
      hidden = vocoder1d::Conv1d(hidden, config.decoder_input_dim, padded_len,
                                 weights.conv_in_weight, &weights.conv_in_bias,
                                 config.decoder_hidden_dim, /*kernel=*/7, /*stride=*/1,
                                 /*dilation=*/1, /*groups=*/1, &produced);
    }
    current = produced;

    int64_t last_output = config.decoder_hidden_dim;
    for (size_t index = 0; index < config.upsampling_ratios.size(); ++index) {
      const int64_t input_dim = config.decoder_hidden_dim >> index;
      const int64_t output_dim = config.decoder_hidden_dim >> (index + 1);
      last_output = output_dim;
      hidden = VocoderBlock(hidden, input_dim, output_dim, current,
                            config.upsampling_ratios[index], weights.blocks[index], &produced);
      current = produced;
    }
    VocoderSnake(hidden, last_output, current, weights.snake_out_alpha);
    {
      profile::Timer timer("vocoder.pad");
      hidden = vocoder1d::Pad1d(hidden, last_output, current, 3, 3, /*replicate=*/false,
                                &padded_len);
    }
    {
      profile::Timer timer("vocoder.conv1d");
      hidden = vocoder1d::Conv1d(hidden, last_output, padded_len, weights.conv_out_weight,
                                 &weights.conv_out_bias, /*out_channels=*/1, /*kernel=*/7,
                                 /*stride=*/1, /*dilation=*/1, /*groups=*/1, &produced);
    }
    current = produced;
    if (current != length * hop) {
      Fail("MiniMax-Music3 vocoder: " + std::to_string(length) + " latent frames decoded to " +
           std::to_string(current) + " samples, the upsampling ratios imply " +
           std::to_string(length * hop));
    }
    {
      profile::Timer timer("vocoder.tanh");
      for (float& value : hidden) value = static_cast<float>(std::tanh(value));
    }
    if (channel == 0) {
      samples = current;
      waveform.resize(static_cast<size_t>(2 * samples));
    }
    for (int64_t t = 0; t < current; ++t) {
      waveform[static_cast<size_t>(channel * samples + t)] = hidden[static_cast<size_t>(t)];
    }
  }
  *out_samples = samples;
  return waveform;
}

}  // namespace music3
}  // namespace models
}  // namespace vllm
