// LTX-2.5 DURATION HEAD — see
// include/vllm/model_executor/models/ltx2_duration_head.h for the upstream
// mapping and the four things that fail silently.
//
// The pooler's attention routes through `vt::AttentionCross`, the shared
// non-causal cross-attention seam phase L2 added, rather than through a local
// softmax loop: the queries are the learnable tokens and the keys/values are the
// token stream, so Tq != S and `vt::Attention` cannot express it. Every
// projection routes through `vt::MatmulBT`.
#include "vllm/model_executor/models/ltx2_duration_head.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

#include "vllm/model_executor/model_loader/safetensors_reader.h"
#include "vt/backend.h"
#include "vt/dtype.h"
#include "vt/ops.h"
#include "vt/tensor.h"

namespace vllm {
namespace {

[[noreturn]] void Refuse(const std::string& message) { throw std::runtime_error(message); }

void Require(bool condition, const std::string& message) {
  if (!condition) Refuse(message);
}

// `torch.nn.Linear`: out = in @ weight^T + bias, with `weight` [out, in].
std::vector<float> Linear(vt::Queue& q, const float* in, int64_t rows, int64_t in_features,
                          const std::vector<float>& weight, const std::vector<float>& bias,
                          int64_t out_features) {
  Require(weight.size() == static_cast<size_t>(out_features * in_features),
          "ltx2 duration head: linear weight has the wrong element count");
  std::vector<float> out(static_cast<size_t>(rows * out_features));
  vt::Tensor a = vt::Tensor::Contiguous(const_cast<float*>(in), vt::DType::kF32, vt::Device{},
                                        {rows, in_features});
  vt::Tensor w = vt::Tensor::Contiguous(const_cast<float*>(weight.data()), vt::DType::kF32,
                                        vt::Device{}, {out_features, in_features});
  vt::Tensor o = vt::Tensor::Contiguous(out.data(), vt::DType::kF32, vt::Device{},
                                        {rows, out_features});
  vt::MatmulBT(q, o, a, w);
  for (int64_t r = 0; r < rows; ++r) {
    for (int64_t i = 0; i < out_features; ++i) {
      out[static_cast<size_t>(r * out_features + i)] += bias[static_cast<size_t>(i)];
    }
  }
  return out;
}

// `torch.nn.functional.gelu(x, approximate="tanh")` — the same activation the
// DiT's FeedForward uses (gelu_approx.py:4-10).
//
// POINTWISE f64, WIDER than upstream's f32, and NOT covered by the suite's
// f64-reduction convention: this is not a reduction at all. Upstream rounds to
// f32 at each step of the expression, so computing the whole thing in double and
// rounding once is numerically FINER than the mirror rather than equal to it —
// the too-wide polarity AGENTS.md warns about, which a value gate cannot catch.
// Left as-is here rather than narrowed in a review-repair branch, because
// narrowing moves the duration-head goldens and so owes its own red-first change.
// Same class as `Silu` in ltx2_upsampler.cpp; see that file's header note.
float GeluTanh(float x) {
  const double v = static_cast<double>(x);
  const double inner = 0.7978845608028654 * (v + 0.044715 * v * v * v);
  return static_cast<float>(0.5 * v * (1.0 + std::tanh(inner)));
}

}  // namespace

std::vector<Ltx2DurationHeadTensorSpec> EnumerateLtx2DurationHeadTensors(
    const Ltx2DurationHeadConfig& config) {
  // torch's `named_parameters()` yields a module's own bare nn.Parameters BEFORE
  // it descends into submodules, so the two modality embeddings come first even
  // though they are declared after `video_input_proj`.
  const std::string p = config.prefix;
  const int64_t hidden = config.pooler_hidden_dim;
  std::vector<Ltx2DurationHeadTensorSpec> specs;
  specs.push_back({p + "video_modality_emb", {hidden}});
  specs.push_back({p + "audio_modality_emb", {hidden}});
  specs.push_back({p + "video_input_proj.weight", {hidden, config.video_cross_attention_dim}});
  specs.push_back({p + "video_input_proj.bias", {hidden}});
  specs.push_back({p + "audio_input_proj.weight", {hidden, config.audio_cross_attention_dim}});
  specs.push_back({p + "audio_input_proj.bias", {hidden}});
  specs.push_back({p + "attention_pooler.query_tokens", {config.num_queries, hidden}});
  // nn.MultiheadAttention's PACKED projection: [3 * E, E] in Q, K, V order.
  specs.push_back({p + "attention_pooler.cross_attn.in_proj_weight", {3 * hidden, hidden}});
  specs.push_back({p + "attention_pooler.cross_attn.in_proj_bias", {3 * hidden}});
  specs.push_back({p + "attention_pooler.cross_attn.out_proj.weight", {hidden, hidden}});
  specs.push_back({p + "attention_pooler.cross_attn.out_proj.bias", {hidden}});
  specs.push_back({p + "mlp_hidden.weight", {config.mlp_hidden, hidden * config.num_queries}});
  specs.push_back({p + "mlp_hidden.bias", {config.mlp_hidden}});
  specs.push_back({p + "mlp_out.weight", {1, config.mlp_hidden}});
  specs.push_back({p + "mlp_out.bias", {1}});
  return specs;
}

std::vector<float> Ltx2DurationAttentionPool(const Ltx2DurationHeadConfig& config,
                                             const Ltx2VaeWeights& weights, const float* tokens,
                                             int64_t batch, int64_t token_count) {
  Require(tokens != nullptr, "ltx2 duration pooler: `tokens` is required");
  const int64_t hidden = config.pooler_hidden_dim;
  const int64_t heads = config.num_pooler_heads;
  Require(hidden % heads == 0,
          "ltx2 duration pooler: pooler_hidden_dim " + std::to_string(hidden) +
              " must be divisible by num_pooler_heads " + std::to_string(heads));
  const int64_t head_dim = hidden / heads;
  const int64_t queries = config.num_queries;
  const std::string p = config.prefix + "attention_pooler.";

  vt::Queue q{vt::Device{}, nullptr};
  const std::vector<float>& in_proj_weight = weights.Get(p + "cross_attn.in_proj_weight");
  const std::vector<float>& in_proj_bias = weights.Get(p + "cross_attn.in_proj_bias");
  Require(in_proj_weight.size() == static_cast<size_t>(3 * hidden * hidden),
          "ltx2 duration pooler: in_proj_weight must be [3 * E, E]");

  // The packed [3E, E] projection, sliced Q / K / V — and it is CROSS attention,
  // so the three slices are applied to two DIFFERENT inputs. A port that treated
  // it as self-attention would project the queries three times.
  auto slice_weight = [&](int64_t index) {
    return std::vector<float>(
        in_proj_weight.begin() + index * hidden * hidden,
        in_proj_weight.begin() + (index + 1) * hidden * hidden);
  };
  auto slice_bias = [&](int64_t index) {
    return std::vector<float>(in_proj_bias.begin() + index * hidden,
                              in_proj_bias.begin() + (index + 1) * hidden);
  };

  // The learnable queries, broadcast across the batch (duration_head.py:47).
  const std::vector<float>& query_tokens = weights.Get(p + "query_tokens");
  std::vector<float> expanded(static_cast<size_t>(batch * queries * hidden));
  for (int64_t b = 0; b < batch; ++b) {
    std::copy(query_tokens.begin(), query_tokens.end(),
              expanded.begin() + b * queries * hidden);
  }

  const std::vector<float> qw = slice_weight(0);
  const std::vector<float> kw = slice_weight(1);
  const std::vector<float> vw = slice_weight(2);
  const std::vector<float> qb = slice_bias(0);
  const std::vector<float> kb = slice_bias(1);
  const std::vector<float> vb = slice_bias(2);

  std::vector<float> qp =
      Linear(q, expanded.data(), batch * queries, hidden, qw, qb, hidden);
  std::vector<float> kp = Linear(q, tokens, batch * token_count, hidden, kw, kb, hidden);
  std::vector<float> vp = Linear(q, tokens, batch * token_count, hidden, vw, vb, hidden);

  std::vector<float> attn(static_cast<size_t>(batch * queries * hidden));
  // torch SDPA's default scale is `E ** -0.5` with E = head_dim.
  vt::AttentionCrossArgs args;
  args.scale = static_cast<float>(1.0 / std::sqrt(static_cast<double>(head_dim)));
  for (int64_t b = 0; b < batch; ++b) {
    vt::Tensor tq = vt::Tensor::Contiguous(qp.data() + b * queries * hidden, vt::DType::kF32,
                                           vt::Device{}, {queries, heads, head_dim});
    vt::Tensor tk = vt::Tensor::Contiguous(kp.data() + b * token_count * hidden, vt::DType::kF32,
                                           vt::Device{}, {token_count, heads, head_dim});
    vt::Tensor tv = vt::Tensor::Contiguous(vp.data() + b * token_count * hidden, vt::DType::kF32,
                                           vt::Device{}, {token_count, heads, head_dim});
    vt::Tensor to = vt::Tensor::Contiguous(attn.data() + b * queries * hidden, vt::DType::kF32,
                                           vt::Device{}, {queries, heads, head_dim});
    // No mask, by construction: the connector has already replaced every padded
    // position with a register and zeroed its mask (duration_head.py:15-16).
    vt::AttentionCross(q, to, tq, tk, tv, nullptr, args);
  }

  return Linear(q, attn.data(), batch * queries, hidden,
                weights.Get(p + "cross_attn.out_proj.weight"),
                weights.Get(p + "cross_attn.out_proj.bias"), hidden);
}

std::vector<float> Ltx2DurationPredict(const Ltx2DurationHeadConfig& config,
                                       const Ltx2VaeWeights& weights, const float* video_tokens,
                                       int64_t video_token_count, const float* audio_tokens,
                                       int64_t audio_token_count, int64_t batch) {
  // duration_head.py:104-105 — upstream's own ValueError.
  Require(video_tokens != nullptr || audio_tokens != nullptr,
          "ltx2 duration head: forward requires at least one of video_tokens / audio_tokens");

  const std::string p = config.prefix;
  const int64_t hidden = config.pooler_hidden_dim;
  vt::Queue q{vt::Device{}, nullptr};

  // The modality embedding is added AFTER the projection (:109, :111), which is
  // what lets the pooler tell the two streams apart.
  auto project = [&](const float* tokens, int64_t token_count, const std::string& stream) {
    std::vector<float> out =
        Linear(q, tokens, batch * token_count, stream == "video"
                                                   ? config.video_cross_attention_dim
                                                   : config.audio_cross_attention_dim,
               weights.Get(p + stream + "_input_proj.weight"),
               weights.Get(p + stream + "_input_proj.bias"), hidden);
    const std::vector<float>& emb = weights.Get(p + stream + "_modality_emb");
    for (int64_t row = 0; row < batch * token_count; ++row) {
      for (int64_t i = 0; i < hidden; ++i) {
        out[static_cast<size_t>(row * hidden + i)] += emb[static_cast<size_t>(i)];
      }
    }
    return out;
  };

  std::vector<float> video_projected;
  std::vector<float> audio_projected;
  if (video_tokens != nullptr) video_projected = project(video_tokens, video_token_count, "video");
  if (audio_tokens != nullptr) audio_projected = project(audio_tokens, audio_token_count, "audio");

  const int64_t video_rows = video_tokens != nullptr ? video_token_count : 0;
  const int64_t audio_rows = audio_tokens != nullptr ? audio_token_count : 0;
  const int64_t token_count = video_rows + audio_rows;

  // `torch.cat(token_groups, dim=1)` (:113) — along the TOKEN axis, per batch
  // row. Concatenating along the feature axis also type-checks here.
  std::vector<float> tokens(static_cast<size_t>(batch * token_count * hidden));
  for (int64_t b = 0; b < batch; ++b) {
    float* dst = tokens.data() + b * token_count * hidden;
    if (video_rows > 0) {
      std::copy(video_projected.begin() + b * video_rows * hidden,
                video_projected.begin() + (b + 1) * video_rows * hidden, dst);
    }
    if (audio_rows > 0) {
      std::copy(audio_projected.begin() + b * audio_rows * hidden,
                audio_projected.begin() + (b + 1) * audio_rows * hidden,
                dst + video_rows * hidden);
    }
  }

  const std::vector<float> pooled =
      Ltx2DurationAttentionPool(config, weights, tokens.data(), batch, token_count);
  // `pooled.reshape(pooled.shape[0], -1)` (:115) — every query's vector,
  // concatenated, which is why mlp_hidden's input width is hidden * num_queries.
  const std::vector<float> hidden_out =
      Linear(q, pooled.data(), batch, hidden * config.num_queries,
             weights.Get(p + "mlp_hidden.weight"), weights.Get(p + "mlp_hidden.bias"),
             config.mlp_hidden);
  std::vector<float> activated(hidden_out.size());
  for (size_t i = 0; i < hidden_out.size(); ++i) activated[i] = GeluTanh(hidden_out[i]);

  const std::vector<float> log_duration =
      Linear(q, activated.data(), batch, config.mlp_hidden, weights.Get(p + "mlp_out.weight"),
             weights.Get(p + "mlp_out.bias"), 1);

  // :117-118 — the regression is trained in LOG-seconds and exponentiated here,
  // so callers always get seconds. Returning the raw regression gives a finite,
  // plausible, completely different duration.
  std::vector<float> seconds(static_cast<size_t>(batch));
  for (int64_t b = 0; b < batch; ++b) {
    seconds[static_cast<size_t>(b)] =
        static_cast<float>(std::exp(static_cast<double>(log_duration[static_cast<size_t>(b)])));
  }
  return seconds;
}


// ─── THE DRIVER (row LTX25-DURATION-HEAD-WIRE, #2900) ────────────────────────
//
// Everything above this line is the head's ARITHMETIC. Everything below is what
// upstream wraps around it to turn a predicted duration into the frame count a
// render actually uses, and it is where the silent failures live: the forward
// returning a plausible number is not the same as the pipeline asking for the
// right length of video.

int64_t Ltx2SnapFramesToGrid(int64_t frames, int64_t time_scale) {
  // `snap_frames_to_grid` (utils/helpers.py:554-562). Upstream refuses `< 1`
  // (:560-561) rather than returning a degenerate count, and that refusal is
  // also what makes the C++ division safe: Python's `//` FLOORS and C++'s `/`
  // truncates toward zero, and the two agree only for a non-negative numerator.
  // With `frames >= 1` the numerator is `frames - 1 >= 0`, so they agree here —
  // and the guard is load-bearing rather than defensive.
  Require(frames >= 1,
          "ltx2 duration: frames must be >= 1, got " + std::to_string(frames) +
              " (snap_frames_to_grid, utils/helpers.py:560-561)");
  Require(time_scale >= 1,
          "ltx2 duration: the VAE temporal scale must be >= 1, got " + std::to_string(time_scale));
  return ((frames - 1) / time_scale) * time_scale + 1;
}

namespace {

// `round()` AS PYTHON DEFINES IT, which is half-to-EVEN and NOT `std::llround`.
//
// This is the one rule here whose obvious C++ spelling is wrong, and it is not a
// measure-zero curiosity. `std::llround` rounds a half AWAY FROM ZERO, so
// 0.34 s at 25 fps — exactly 8.5 — becomes 9 and snaps to frame 9, where
// upstream takes the even 8 and snaps to frame 1. Eight frames, on a request a
// user can type. `tests/vllm/models/ltx2_duration_wire_goldens.inc` carries that
// case and 0.5 s at 49 fps beside it, with `llround`'s answer in the
// `rejected_half_away` column, so a future edit to this function cannot quietly
// adopt the wrong one.
int64_t RoundHalfToEven(double value) {
  const double floored = std::floor(value);
  const double diff = value - floored;
  if (diff > 0.5) return static_cast<int64_t>(floored) + 1;
  if (diff < 0.5) return static_cast<int64_t>(floored);
  // Exactly a half: take the even neighbour.
  const int64_t low = static_cast<int64_t>(floored);
  return (low % 2 == 0) ? low : low + 1;
}

}  // namespace

int64_t Ltx2SecondsToClampedNumFrames(double seconds, double frame_rate, int64_t min_frames,
                                      int64_t max_frames, int64_t time_scale) {
  // `seconds_to_clamped_num_frames` (utils/helpers.py:565-585), and the ORDER of
  // these three steps is the whole function. Two of them commute in neither
  // direction and both orders type-check:
  //
  //  * THE CLAMP PRECEDES THE SNAP (:579-580). Snapping first and clamping after
  //    gives a different count for every prediction whose floored grid point
  //    leaves the window.
  //  * THE UNDERSHOOT REPAIR SNAPS UP, AND IS ITSELF CAPPED (:581-584).
  //    `snap_frames_to_grid` FLOORS, so a `min_frames` that is not already a
  //    grid point produces a count BELOW the window the caller asked for.
  //    Upstream recovers with a ceiling division back onto the grid and then
  //    takes `min` with `max_frames`, so a degenerate window cannot be broken in
  //    the other direction. That `min` is why a `min_frames == max_frames == 5`
  //    request legitimately returns 5, which is NOT on the 8k+1 grid: upstream
  //    honours the window over the grid when the two cannot both hold.
  //  * A `min_frames < 1` IS NOT REFUSED, because upstream does not refuse it.
  //    `seconds_to_clamped_num_frames(3.0, frame_rate=25.0, min_frames=0,
  //    max_frames=500)` returns 73 at the pin. What upstream DOES raise on is a
  //    clamped count of zero — `(0.005, 25.0, min_frames=0, max_frames=500)`
  //    gives `ValueError: frames must be >= 1, got 0` — and that raise comes out
  //    of `snap_frames_to_grid` below, not out of a bound check here. Refusing
  //    the bound instead moved the refusal to a request upstream serves.
  Require(frame_rate > 0.0, "ltx2 duration: the frame rate must be positive");
  Require(max_frames >= min_frames,
          "ltx2 duration: max_frames (" + std::to_string(max_frames) +
              ") must be >= min_frames (" + std::to_string(min_frames) + ")");
  int64_t raw = RoundHalfToEven(seconds * frame_rate);
  if (raw < min_frames) raw = min_frames;
  if (raw > max_frames) raw = max_frames;
  int64_t frames = Ltx2SnapFramesToGrid(raw, time_scale);
  // The repair branch needs `min_frames >= 2` to fire at all, since the snap
  // returns at least 1. That is what keeps the ceiling division below correct:
  // `(a + b - 1) / b` is `ceil` only for a non-negative `a`, and `min_frames - 1`
  // is `>= 1` wherever this runs. Upstream's `-(-x // ts)` floors on the way in
  // and needs no such argument.
  if (frames < min_frames) {
    const int64_t up = ((min_frames - 1 + time_scale - 1) / time_scale) * time_scale + 1;
    frames = up < max_frames ? up : max_frames;
  }
  return frames;
}

int64_t Ltx2DurationPredictFrames(const Ltx2DurationHeadConfig& config,
                                  const Ltx2VaeWeights& weights, const float* video_tokens,
                                  int64_t video_token_count, const float* audio_tokens,
                                  int64_t audio_token_count, double frame_rate, double min_seconds,
                                  double max_seconds, int64_t time_scale,
                                  float* predicted_seconds) {
  // `DurationPredictor.__call__` (utils/blocks.py:850-889).
  //
  // UPSTREAM'S SINGLE-ITEM-BATCH REFUSAL (:857-861) IS EXPRESSED AS A CONTRACT
  // RATHER THAN A CHECK: this signature has no `batch`, so a caller cannot ask
  // for the shape upstream would reject. The refusal it replaces exists because
  // `.item()` on a multi-row prediction throws a shape error deep in torch; here
  // there is no such row to have.
  // `min_seconds <= 0` is upstream's to serve, not this port's to refuse:
  // `AutoDuration(min_seconds=0.0, max_seconds=20.0)` constructs at the pin
  // (utils/args.py:117-122 checks only `min > max`). The retained bound is that
  // same `min > max`, which IS upstream's.
  Require(max_seconds >= min_seconds,
          "ltx2 duration: max_seconds must be >= min_seconds");
  const std::vector<float> seconds_pred =
      Ltx2DurationPredict(config, weights, video_tokens, video_token_count, audio_tokens,
                          audio_token_count, /*batch=*/1);
  Require(seconds_pred.size() == 1, "ltx2 duration: the head returned more than one prediction");
  const double seconds = static_cast<double>(seconds_pred[0]);
  if (predicted_seconds != nullptr) *predicted_seconds = seconds_pred[0];
  // `min_frames` / `max_frames` are `round(bound * frame_rate)` (:872-873), and
  // they go through the SAME rounding as the prediction: a bound converted with
  // a different rule would put the window in a different place from the value it
  // is clamping.
  const int64_t min_frames = RoundHalfToEven(min_seconds * frame_rate);
  const int64_t max_frames = RoundHalfToEven(max_seconds * frame_rate);
  // NO `min_frames < 1 ? 1 : min_frames` FLOOR. Upstream passes the rounded bound
  // through untouched (blocks.py:872-875), so a `min_frames` of 0 with a
  // prediction that rounds to 0 frames RAISES — `seconds_to_clamped_num_frames
  // (0.005, frame_rate=25.0, min_frames=0, max_frames=500)` is `ValueError:
  // frames must be >= 1, got 0` at the pin. Flooring the bound to 1 turned that
  // raise into a silent single frame.
  return Ltx2SecondsToClampedNumFrames(seconds, frame_rate, min_frames, max_frames, time_scale);
}

void Ltx2RequireNumFramesSource(bool auto_requested, bool has_predictor) {
  // `require_num_frames_source` (utils/blocks.py:894-905). The MESSAGE is
  // upstream's; the POSITION — the very top of `__call__`, before prompt
  // encoding or any other work — is the caller's obligation and is what
  // `test_ltx2_video`'s ordering case gates.
  if (auto_requested && !has_predictor) {
    Refuse(
        "num_frames was AutoDuration but this checkpoint has no DurationHead weights to "
        "auto-predict duration from (DurationHead ships from LTX 2.5 / gemma4 onward). "
        "Pass num_frames explicitly.");
  }
}

namespace {

std::string ShapeText(const std::vector<int64_t>& shape) {
  std::string out = "[";
  for (size_t i = 0; i < shape.size(); ++i) {
    if (i != 0) out += ", ";
    out += std::to_string(shape[i]);
  }
  return out + "]";
}

// The head's file is a PLAIN safetensors bag -- no quantization plan, no shards
// -- so this reads it directly rather than through the DiT planner. Two dtypes
// are admitted because a checkpoint may store the head at either; anything else
// refuses by name rather than reinterpreting bytes.
void ReadTensorBytes(const StTensor& t, const std::string& name, int64_t numel, bool want_bf16,
                     void* out) {
  const bool src_bf16 = t.dtype == "BF16";
  if (!src_bf16 && t.dtype != "F32") {
    Refuse("ltx2 duration head: '" + name + "' is stored as " + t.dtype +
           ", and the head reads F32 or BF16. The FP8 and NVFP4 arms are upstream's "
           "quantization policies (quantization_factory.py:22-26) and are not implemented "
           "for this component");
  }
  const size_t elem = src_bf16 ? sizeof(uint16_t) : sizeof(float);
  if (t.nbytes != static_cast<size_t>(numel) * elem) {
    Refuse("ltx2 duration head: '" + name + "' holds " + std::to_string(t.nbytes) +
           " bytes, and its shape needs " + std::to_string(static_cast<size_t>(numel) * elem));
  }
  for (int64_t i = 0; i < numel; ++i) {
    const size_t at = static_cast<size_t>(i);
    float value;
    if (src_bf16) {
      uint16_t word = 0;
      std::memcpy(&word, t.data + at * sizeof(uint16_t), sizeof(word));
      value = vt::BF16ToF32(word);
    } else {
      std::memcpy(&value, t.data + at * sizeof(float), sizeof(value));
    }
    if (want_bf16) {
      // A BF16 source is moved WORD FOR WORD rather than round-tripped: widening
      // and re-narrowing is lossless for bf16 but says the wrong thing about what
      // this arm does, and it would hide a real narrowing behind an identity.
      uint16_t word;
      if (src_bf16) {
        std::memcpy(&word, t.data + at * sizeof(uint16_t), sizeof(word));
      } else {
        word = vt::F32ToBF16(value);
      }
      static_cast<uint16_t*>(out)[at] = word;
    } else {
      static_cast<float*>(out)[at] = value;
    }
  }
}

}  // namespace

bool Ltx2LoadDurationHeadWeights(const SafetensorsFile& file,
                                 const Ltx2DurationHeadConfig& config, vt::DType compute_dtype,
                                 Ltx2VaeWeights* out) {
  // `DurationPredictor.from_checkpoint` (utils/blocks.py:816-848), and the
  // RETURN TYPE is the mirror. Upstream returns `None` rather than raising when
  // the file carries no head, because every checkpoint predating LTX-2.5 /
  // gemma4 IS that file; refusing would reject checkpoints upstream runs.
  //
  // A PARTIAL HEAD IS `None` TOO. Upstream builds with `strict=False`, so a
  // missing tensor leaves its parameter on the meta device, and `:838` asks
  // `any(param.is_meta ...)` and returns `None` for the whole head. Fourteen of
  // fifteen tensors is therefore no predictor, not a predictor that faults in
  // the forward.
  //
  // A TENSOR THAT IS PRESENT AT THE WRONG SHAPE IS THE ONE CASE THAT REFUSES,
  // and the distinction is upstream's rather than invented here: nothing is
  // missing, so `any(param.is_meta)` is False and the Builder would have had to
  // bind it. Binding it runs the head as a silently different module.
  Require(out != nullptr, "ltx2 duration head: `out` is required");
  Require(compute_dtype == vt::DType::kF32 || compute_dtype == vt::DType::kBF16,
          "ltx2 duration head: the weights can be materialized as f32 (the parity arm) or bf16 "
          "(upstream's own model dtype, distilled.py:109). No other arm is implemented");
  const std::vector<Ltx2DurationHeadTensorSpec> specs = EnumerateLtx2DurationHeadTensors(config);
  const std::vector<std::string>& names = file.Names();
  size_t present = 0;
  for (const Ltx2DurationHeadTensorSpec& spec : specs) {
    if (std::find(names.begin(), names.end(), spec.name) != names.end()) ++present;
  }
  // Absent AND partial are both upstream's `None`.
  if (present != specs.size()) return false;

  Ltx2VaeWeights bag;
  bag.dtype = compute_dtype;
  for (const Ltx2DurationHeadTensorSpec& spec : specs) {
    const StTensor& t = file.Get(spec.name);
    if (t.shape != spec.shape) {
      Refuse("ltx2 duration head: '" + spec.name + "' is " + ShapeText(t.shape) +
             " in the checkpoint but the head configuration needs " + ShapeText(spec.shape) +
             ". The two describe different modules; refusing rather than binding a tensor "
             "whose shape would run the head as something else");
    }
    int64_t numel = 1;
    for (const int64_t d : spec.shape) numel *= d;
    // The KEY the bag is filled under is the spec's own name, prefix included,
    // because `Ltx2DurationPredict` looks the tensors up through the same
    // `config.prefix`. Upstream's `DURATION_HEAD_KEY_OPS` strips the prefix on
    // the way in and the module then holds bare names; the port carries the
    // prefix through the config instead, which is the same rename done once.
    if (compute_dtype == vt::DType::kBF16) {
      std::vector<uint16_t> narrow(static_cast<size_t>(numel));
      ReadTensorBytes(t, spec.name, numel, /*want_bf16=*/true, narrow.data());
      bag.bf16[spec.name] = std::move(narrow);
    } else {
      std::vector<float> widened(static_cast<size_t>(numel));
      ReadTensorBytes(t, spec.name, numel, /*want_bf16=*/false, widened.data());
      bag.tensors[spec.name] = std::move(widened);
    }
  }
  *out = std::move(bag);
  return true;
}

}  // namespace vllm
