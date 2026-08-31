// Qwen4-Exp W2 — host reference for the two components with no vLLM op.
// Issue #1987, spec `.agents/specs/qwen4-exp-flash-next.md`. The header carries
// the scope statement, the upstream anchors, the batching seam and the reason
// this is host code; read it first.
//
// Oracle: huggingface/transformers v5.16.0 (this row's accepted lane pin). Line
// citations below are `modeling_qwen4_exp.py` at that tag unless stated.

#include "vllm/model_executor/models/qwen4_exp_ple.h"

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>

namespace vllm::qwen4_exp {
namespace {

constexpr uint64_t kSplitmixGamma = 0x9E3779B97F4A7C15ULL;  // :973
constexpr uint64_t kSplitmixM1 = 0xBF58476D1CE4E5B9ULL;     // :974
constexpr uint64_t kSplitmixM2 = 0x94D049BB133111EBULL;     // :975
constexpr uint64_t kPrime1 = 10007ULL;                      // :976

[[noreturn]] void Refuse(const std::string& what) {
  throw std::invalid_argument("qwen4_exp PLE: " + what);
}

// The SECOND half of the int64 bound, and the half that has no `input_ids` loop
// to catch it. `eos_token_id` is the one id in the n-gram mix that does not
// come from the caller's tokens: `Reset` seeds the history with it and
// `ShiftRightIgnoreEos` emits it at every segment start, so it lands on the
// FIRST TOKEN OF EVERY SEQUENCE. Upstream needs no check because
// `config.eos_token_id` cannot be out of range (:1032 even unwraps a LIST);
// `PleGeometry` is a plain struct whose eos DEFAULTS TO -1, and at -1 the
// int64 product goes negative, where `torch.remainder` normalises it and our
// `static_cast<uint64_t>` reinterprets the same bits near 2^64. Measured
// against transformers v5.16.0: upstream row 0 `[2, 35, 67, 96]` against ours
// `[8, 30, 67, 96]`, with no exception and no shape change.
void RefuseBadEos(const PleGeometry& geom) {
  if (geom.eos_token_id < 0 || geom.eos_token_id >= geom.vocab_size) {
    Refuse("eos_token_id " + std::to_string(geom.eos_token_id) +
           " is outside [0, vocab_size=" + std::to_string(geom.vocab_size) +
           "); it enters the n-gram mix at every segment start, so it would "
           "overflow int64 and diverge in silence exactly like an "
           "out-of-range input id");
  }
}

// `Qwen4ExpTextRMSNorm._norm` + `.forward`, :167-178, group_size arm.
// Two traps in six lines: the reduction is over the GROUP, not the row, and the
// scale is `(1.0 + weight)` rather than `weight`, so a zeroed buffer is
// identity. Upstream promotes to float32 first; we accumulate in double, which
// is a reference choice and not a divergence — the tolerance in the gate covers
// the reduction-order difference against torch.
//
// THE OTHER `GroupedRmsNorm` IN THIS NAMESPACE HAS THE OPPOSITE POLARITY.
// `vllm::qwen4_exp::GroupedRmsNorm` (`qwen4_exp_hc.cpp:52`, declared in
// `qwen4_exp_hc.h`) applies `out * weight` on vLLM's FOLDED gamma, which callers
// build with `HcNormWeightFromHf`; this one adds the 1 itself and takes the raw
// HuggingFace parameter. This function keeps internal linkage and this file does
// not include `qwen4_exp_hc.h`, so the two cannot be confused by the compiler --
// only by a reader, and a reader confusing two gamma conventions is #2218
// exactly. Whichever of these survives PLE's move onto a standalone grouped-norm
// `vt::` op (`## Owed` item 1) should be the only one.
void GroupedRmsNorm(const float* x, int64_t rows, int64_t width,
                    int64_t group_size, double eps, const float* weight,
                    float* out) {
  const int64_t groups = width / group_size;
  for (int64_t r = 0; r < rows; ++r) {
    const float* xr = x + r * width;
    float* outr = out + r * width;
    for (int64_t g = 0; g < groups; ++g) {
      const int64_t base = g * group_size;
      double sumsq = 0.0;
      for (int64_t i = 0; i < group_size; ++i) {
        const double v = static_cast<double>(xr[base + i]);
        sumsq += v * v;
      }
      const double scale = 1.0 / std::sqrt(sumsq / static_cast<double>(group_size) + eps);
      for (int64_t i = 0; i < group_size; ++i) {
        outr[base + i] = static_cast<float>(static_cast<double>(xr[base + i]) * scale *
                                            (1.0 + static_cast<double>(weight[base + i])));
      }
    }
  }
}

// `nn.Linear(bias=False)`: y[r, o] = sum_i x[r, i] * w[o, i].
void Linear(const float* x, const float* w, int64_t rows, int64_t in_features,
            int64_t out_features, float* y) {
  for (int64_t r = 0; r < rows; ++r) {
    const float* xr = x + r * in_features;
    float* yr = y + r * out_features;
    for (int64_t o = 0; o < out_features; ++o) {
      const float* wo = w + o * in_features;
      double acc = 0.0;
      for (int64_t i = 0; i < in_features; ++i) {
        acc += static_cast<double>(xr[i]) * static_cast<double>(wo[i]);
      }
      yr[o] = static_cast<float>(acc);
    }
  }
}

double Sigmoid(double v) { return 1.0 / (1.0 + std::exp(-v)); }

double Silu(double v) { return v * Sigmoid(v); }

}  // namespace

// :979-983. Unsigned throughout — divergence site #1.
uint64_t SplitMix64(uint64_t value) {
  value = value + kSplitmixGamma;
  value = (value ^ (value >> 30)) * kSplitmixM1;
  value = (value ^ (value >> 27)) * kSplitmixM2;
  return value ^ (value >> 31);
}

// :998-1006.
bool IsPrime(int64_t value) {
  if (value < 2) return false;
  if (value % 2 == 0) return value == 2;
  for (int64_t divisor = 3; divisor * divisor <= value; divisor += 2) {
    if (value % divisor == 0) return false;
  }
  return true;
}

// :1009-1015.
int64_t FindNthPrimeAfter(int64_t start, int64_t count) {
  int64_t prime = start;
  for (int64_t i = 0; i < count; ++i) {
    ++prime;
    while (!IsPrime(prime)) ++prime;
  }
  return prime;
}

// :986-995. The modulo is UNSIGNED — divergence site #2. `SplitMix64` routinely
// returns a value above 2^63 and a signed `%` there yields a negative residue,
// which then becomes an even multiplier and a negative row index.
std::vector<int64_t> BuildLayerMultipliers(int64_t unigram_vocab_size,
                                           int64_t ngram_size,
                                           int64_t ple_layer_index,
                                           int64_t seed) {
  if (ngram_size <= 0) Refuse("ngram_size must be positive");
  const uint64_t max_long = static_cast<uint64_t>(INT64_MAX);
  const uint64_t divisor =
      unigram_vocab_size > 1 ? static_cast<uint64_t>(unigram_vocab_size) : 1ULL;
  const uint64_t multiplier_max = max_long / divisor;
  const uint64_t half_bound = multiplier_max / 2 > 0 ? multiplier_max / 2 : 1ULL;
  const uint64_t base_seed =
      static_cast<uint64_t>(seed) + kPrime1 * static_cast<uint64_t>(ple_layer_index);

  std::vector<int64_t> multipliers;
  multipliers.reserve(static_cast<size_t>(ngram_size));
  for (int64_t index = 0; index < ngram_size; ++index) {
    const uint64_t value =
        base_seed + kSplitmixGamma * static_cast<uint64_t>(index + 1);
    multipliers.push_back(
        static_cast<int64_t>(2ULL * (SplitMix64(value) % half_bound) + 1ULL));
  }
  return multipliers;
}

// :1019-1051. The offsets are an exclusive prefix sum over the head vocab
// sizes, and `padded_vocab_size` rounds UP: the released config leaves 90
// unaddressable rows (320001446 -> 320001536).
NGramTableLayout BuildNGramTableLayout(const PleGeometry& geom,
                                       int64_t ple_layer_index) {
  const int64_t heads = geom.ngram_heads();
  if (heads <= 0) Refuse("ngram_heads must be positive");
  if (geom.ple_embed_dim <= 0 || geom.ple_embed_dim % heads != 0) {
    Refuse("ple_embed_dim must be positive and divisible by ngram_heads");
  }
  if (geom.make_ngram_vocab_size_divisible_by <= 0) {
    Refuse("make_ngram_vocab_size_divisible_by must be positive");
  }
  RefuseBadEos(geom);

  NGramTableLayout layout;
  layout.head_vocab_sizes.reserve(static_cast<size_t>(heads));
  layout.head_offsets.reserve(static_cast<size_t>(heads));
  for (int64_t head_idx = 0; head_idx < heads; ++head_idx) {
    const int64_t global_head_idx = ple_layer_index * heads + head_idx;
    const int64_t size =
        FindNthPrimeAfter(geom.ngram_vocab_size_base - 1, global_head_idx + 1);
    layout.head_vocab_sizes.push_back(size);
    layout.head_offsets.push_back(layout.total_vocab_size);
    layout.total_vocab_size += size;
  }
  layout.layer_multipliers = BuildLayerMultipliers(
      geom.vocab_size, geom.ngram_size, ple_layer_index, geom.seed);

  const int64_t divisor = geom.make_ngram_vocab_size_divisible_by;
  layout.padded_vocab_size =
      ((layout.total_vocab_size + divisor - 1) / divisor) * divisor;
  return layout;
}

void PleSequenceState::Reset(const PleGeometry& geom) {
  conv.assign(static_cast<size_t>(geom.stream_width() * geom.short_conv_state_len()),
              0.0F);
  tokens.assign(static_cast<size_t>(geom.context_len()), geom.eos_token_id);
}

// :1053-1067.
void ShiftRightIgnoreEos(const int64_t* token_ids, int64_t seq_len, int64_t shift,
                         int64_t eos_token_id, int64_t* out) {
  if (shift == 0) {
    for (int64_t i = 0; i < seq_len; ++i) out[i] = token_ids[i];
    return;
  }
  // `previous_eos` is the running max of EOS positions STRICTLY BEFORE i, which
  // is why an EOS token belongs to the segment it terminates rather than to the
  // one it opens.
  int64_t previous_eos = -1;
  for (int64_t i = 0; i < seq_len; ++i) {
    const int64_t segment_start = previous_eos + 1;
    const int64_t position_in_segment = i - segment_start;
    const int64_t source = i - shift;
    out[i] = (position_in_segment >= shift && source >= 0) ? token_ids[source]
                                                           : eos_token_id;
    if (token_ids[i] == eos_token_id) previous_eos = i;
  }
}

// :1069-1112 for the ids, plus cache_utils.py:1037-1075 for state 2.
//
// Upstream's state-2 dance reduces to one line — `tokens := last context_len of
// (tokens ++ input_ids)` — and the reduction is worth stating because the two
// branches it collapses look different: on the first call it EOS-left-pads a
// short chunk before caching (:1080-1088) precisely to dodge
// `update_conv_state`'s zero pad, and on later calls it concatenates. With
// `tokens` EOS-seeded by `Reset` both branches produce the same bytes.
void BuildNGramIds(const PleGeometry& geom, const NGramTableLayout& layout,
                   const int64_t* input_ids, int64_t num_tokens,
                   PleSequenceState* state, int64_t* out_ids) {
  if (state == nullptr) Refuse("BuildNGramIds needs a sequence state");
  const int64_t context_len = geom.context_len();
  const int64_t heads = geom.ngram_heads();
  if (static_cast<int64_t>(state->tokens.size()) != context_len) {
    Refuse("sequence state was not Reset for this geometry");
  }
  if (num_tokens <= 0) return;

  // The bound the whole int64 argument rests on. Upstream cannot be handed an
  // out-of-range id; we can, and the failure is a silent overflow. Both halves
  // are checked here and not only at layout time, because the geometry is a
  // plain struct the caller still owns.
  RefuseBadEos(geom);
  for (int64_t t = 0; t < num_tokens; ++t) {
    if (input_ids[t] < 0 || input_ids[t] >= geom.vocab_size) {
      Refuse("token id " + std::to_string(input_ids[t]) +
             " is outside [0, vocab_size=" + std::to_string(geom.vocab_size) +
             "); the n-gram mix would overflow int64 and diverge in silence");
    }
  }

  const int64_t history_len = context_len + num_tokens;
  std::vector<int64_t> history(static_cast<size_t>(history_len));
  for (int64_t i = 0; i < context_len; ++i) history[i] = state->tokens[i];
  for (int64_t t = 0; t < num_tokens; ++t) history[context_len + t] = input_ids[t];

  std::vector<std::vector<int64_t>> shifted(static_cast<size_t>(geom.ngram_size));
  for (int64_t s = 0; s < geom.ngram_size; ++s) {
    shifted[static_cast<size_t>(s)].resize(static_cast<size_t>(history_len));
    ShiftRightIgnoreEos(history.data(), history_len, s, geom.eos_token_id,
                        shifted[static_cast<size_t>(s)].data());
  }

  // The XOR runs in uint64: every operand is non-negative by the bound checked
  // above, so this is bit-identical to upstream's int64 tensor op and avoids
  // signed-overflow UB on the way.
  for (int64_t ngram = 2; ngram <= geom.ngram_size; ++ngram) {
    const int64_t start_idx = (ngram - 2) * geom.heads_per_ngram;
    for (int64_t i = 0; i < history_len; ++i) {
      const int64_t row = i - context_len;
      if (row < 0) continue;  // upstream builds all rows, then keeps the last T
      uint64_t mixed = static_cast<uint64_t>(shifted[0][static_cast<size_t>(i)]) *
                       static_cast<uint64_t>(layout.layer_multipliers[0]);
      for (int64_t position = 1; position < ngram; ++position) {
        mixed ^= static_cast<uint64_t>(
                     shifted[static_cast<size_t>(position)][static_cast<size_t>(i)]) *
                 static_cast<uint64_t>(layout.layer_multipliers[static_cast<size_t>(position)]);
      }
      for (int64_t h = 0; h < geom.heads_per_ngram; ++h) {
        const int64_t head = start_idx + h;
        const uint64_t size =
            static_cast<uint64_t>(layout.head_vocab_sizes[static_cast<size_t>(head)]);
        out_ids[row * heads + head] =
            static_cast<int64_t>(mixed % size) +
            layout.head_offsets[static_cast<size_t>(head)];
      }
    }
  }

  // Advance state 2.
  for (int64_t i = 0; i < context_len; ++i) {
    state->tokens[static_cast<size_t>(i)] =
        history[static_cast<size_t>(history_len - context_len + i)];
  }
}

// :1181.
float SignedSqrtGate(float gate) {
  if (gate == 0.0F) return 0.0F;  // sign(0) == 0, so the origin maps to zero
  const double magnitude = std::abs(static_cast<double>(gate));
  const double clamped = magnitude < 1e-6 ? 1e-6 : magnitude;
  const double out = std::sqrt(clamped);
  return static_cast<float>(gate > 0.0F ? out : -out);
}

// :1150-1167. The pad-and-slice upstream performs unconditionally collapses to
// "prepend the 9-column state", and the state after the call is the last 9
// columns of that same buffer. Both upstream branches — the zero left-pad on a
// short first chunk and the concatenation on every later chunk — land on those
// bytes, which is why there is no `has_previous_state` test here.
void PleShortConv(const PleGeometry& geom, const float* conv1d_weight,
                  const float* normed, int64_t num_tokens,
                  PleSequenceState* state, float* out) {
  if (state == nullptr) Refuse("PleShortConv needs a sequence state");
  const int64_t width = geom.stream_width();
  const int64_t state_len = geom.short_conv_state_len();
  const int64_t kernel = geom.ple_conv_kernel_size;
  const int64_t dilation = geom.ngram_size;
  if (static_cast<int64_t>(state->conv.size()) != width * state_len) {
    Refuse("sequence state was not Reset for this geometry");
  }
  if (num_tokens <= 0) return;

  // [width, state_len + num_tokens], channel-major: the conv is depthwise, so
  // this is the layout a device arm wants too.
  const int64_t span = state_len + num_tokens;
  std::vector<float> buffer(static_cast<size_t>(width * span));
  for (int64_t c = 0; c < width; ++c) {
    float* row = buffer.data() + c * span;
    for (int64_t s = 0; s < state_len; ++s) row[s] = state->conv[c * state_len + s];
    for (int64_t t = 0; t < num_tokens; ++t) row[state_len + t] = normed[t * width + c];
  }

  for (int64_t c = 0; c < width; ++c) {
    const float* row = buffer.data() + c * span;
    const float* w = conv1d_weight + c * kernel;
    for (int64_t t = 0; t < num_tokens; ++t) {
      double acc = 0.0;
      // k = 0..3 reads lags {9, 6, 3, 0}: `t + k * dilation` against a buffer
      // whose current token sits at `t + state_len`, and `(kernel-1)*dilation
      // == state_len` makes the last tap the current token. Causal by that tap.
      for (int64_t k = 0; k < kernel; ++k) {
        acc += static_cast<double>(w[k]) * static_cast<double>(row[t + k * dilation]);
      }
      out[t * width + c] = static_cast<float>(Silu(acc));
    }
  }

  for (int64_t c = 0; c < width; ++c) {
    const float* row = buffer.data() + c * span;
    for (int64_t s = 0; s < state_len; ++s) {
      state->conv[c * state_len + s] = row[span - state_len + s];
    }
  }
}

// :1169-1189.
void PleForward(const PleGeometry& geom, const NGramTableLayout& layout,
                const PleWeights& weights, const float* hidden_states,
                const int64_t* input_ids, int64_t num_tokens,
                const unsigned char* conv_mask, PleSequenceState* state,
                float* out) {
  if (state == nullptr) Refuse("PleForward needs a sequence state");
  const int64_t hidden = geom.hidden_size;
  const int64_t hc = geom.hc_count;
  const int64_t width = geom.stream_width();
  const int64_t heads = geom.ngram_heads();
  const int64_t head_dim = geom.head_dim_per_ngram();
  const int64_t embed_dim = geom.ple_embed_dim;
  if (num_tokens <= 0) return;

  std::vector<int64_t> ids(static_cast<size_t>(num_tokens * heads));
  BuildNGramIds(geom, layout, input_ids, num_tokens, state, ids.data());

  // The gather: `heads` uncoalesced random rows per token. This loop is the
  // batching seam named in the header.
  std::vector<float> embeddings(static_cast<size_t>(num_tokens * embed_dim));
  for (int64_t t = 0; t < num_tokens; ++t) {
    for (int64_t h = 0; h < heads; ++h) {
      const int64_t row = ids[t * heads + h];
      if (row < 0 || row >= layout.padded_vocab_size) {
        Refuse("n-gram row " + std::to_string(row) + " is outside the padded table");
      }
      const float* src = weights.ngram_embedding + row * head_dim;
      float* dst = embeddings.data() + t * embed_dim + h * head_dim;
      for (int64_t d = 0; d < head_dim; ++d) dst[d] = src[d];
    }
  }

  std::vector<float> key(static_cast<size_t>(num_tokens * width));
  Linear(embeddings.data(), weights.key_proj, num_tokens, embed_dim, width, key.data());
  std::vector<float> key_normed(static_cast<size_t>(num_tokens * width));
  GroupedRmsNorm(key.data(), num_tokens, width, hidden, geom.rms_norm_eps,
                 weights.norm_key, key_normed.data());

  std::vector<float> value(static_cast<size_t>(num_tokens * hidden));
  Linear(embeddings.data(), weights.value_proj, num_tokens, embed_dim, hidden,
         value.data());

  std::vector<float> query_normed(static_cast<size_t>(num_tokens * width));
  GroupedRmsNorm(hidden_states, num_tokens, width, hidden, geom.rms_norm_eps,
                 weights.norm_query, query_normed.data());

  const double inv_scale = 1.0 / std::sqrt(static_cast<double>(hidden));
  std::vector<float> gated_value(static_cast<size_t>(num_tokens * width));
  for (int64_t t = 0; t < num_tokens; ++t) {
    for (int64_t s = 0; s < hc; ++s) {
      const float* k = key_normed.data() + t * width + s * hidden;
      const float* q = query_normed.data() + t * width + s * hidden;
      double dot = 0.0;
      for (int64_t d = 0; d < hidden; ++d) {
        dot += static_cast<double>(k[d]) * static_cast<double>(q[d]);
      }
      const float gate = SignedSqrtGate(static_cast<float>(dot * inv_scale));
      const double weight = Sigmoid(static_cast<double>(gate));
      float* dst = gated_value.data() + t * width + s * hidden;
      const float* v = value.data() + t * hidden;
      for (int64_t d = 0; d < hidden; ++d) {
        dst[d] = static_cast<float>(weight * static_cast<double>(v[d]));
      }
    }
  }

  // THE FORK. `gated_value_normed` is what the conv sees; the skip term added
  // back at the end is the UN-NORMED copy, and the 9-column state holds the
  // NORMED one.
  std::vector<float> gated_value_normed(static_cast<size_t>(num_tokens * width));
  GroupedRmsNorm(gated_value.data(), num_tokens, width, hidden, geom.rms_norm_eps,
                 weights.norm_conv, gated_value_normed.data());

  if (conv_mask != nullptr) {
    for (int64_t t = 0; t < num_tokens; ++t) {
      const float m = conv_mask[t] != 0 ? 1.0F : 0.0F;
      for (int64_t c = 0; c < width; ++c) {
        gated_value[t * width + c] *= m;
        gated_value_normed[t * width + c] *= m;
      }
    }
  }

  std::vector<float> conv_out(static_cast<size_t>(num_tokens * width));
  PleShortConv(geom, weights.conv1d, gated_value_normed.data(), num_tokens, state,
               conv_out.data());
  for (int64_t i = 0; i < num_tokens * width; ++i) {
    out[i] = gated_value[static_cast<size_t>(i)] + conv_out[static_cast<size_t>(i)];
  }
}

}  // namespace vllm::qwen4_exp
