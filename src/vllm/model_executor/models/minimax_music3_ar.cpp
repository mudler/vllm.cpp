// MiniMax-Music3 — the AUTOREGRESSIVE half (W2 + W3 of #672).
// See include/vllm/model_executor/models/minimax_music3_ar.h for what is gated
// here and, more importantly, what is NOT: upstream's AR stage has no greedy
// path, so the committed `rvq_codes.npy` is a seeded sample and is consumed as
// an INPUT by these gates rather than predicted by them.
//
// Upstream anchors, at diffusers PR #14456 head c6da9936:
//   condition_embedder_minimax_music3.py:48-76
//   minimax_music3_rvq_depth_decoder.py:28-142
//   modular_pipelines/minimax_music3/encoders.py:54-142, :202-353
#include "vllm/model_executor/models/minimax_music3_ar.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <regex>
#include <stdexcept>

#include "vllm/model_executor/models/vocoder1d.h"
#include "vt/dtype.h"

namespace vllm {
namespace models {
namespace music3 {
namespace {

[[noreturn]] void Fail(const std::string& what) { throw std::runtime_error(what); }

// ONE op boundary. Accumulation stays in double (torch's bf16 matmul accumulates
// in float32 and its RMSNorm variance in float32); what the dtype decides is the
// width the RESULT is stored at, which is where a bf16 module actually loses
// bits. See the header note for why this is a mirror and not a knob.
inline float Store(double value, ArCompute compute) {
  const float narrowed = static_cast<float>(value);
  return compute == ArCompute::kBFloat16 ? vt::BF16ToF32(vt::F32ToBF16(narrowed)) : narrowed;
}

}  // namespace

// ---------------------------------------------------------------------------
// Prompt assembly
// ---------------------------------------------------------------------------

std::string CleanCaption(const std::string& caption) {
  // encoders.py:60 — `<|k v|>` becomes "k is v"; a tag with no space keeps its
  // inner text with no rewrite.
  static const std::regex kSpecialTag(R"(<\|([^|]*)\|>)");
  std::string text;
  auto begin = std::sregex_iterator(caption.begin(), caption.end(), kSpecialTag);
  auto end = std::sregex_iterator();
  size_t last = 0;
  for (auto it = begin; it != end; ++it) {
    const std::smatch& match = *it;
    text.append(caption, last, static_cast<size_t>(match.position(0)) - last);
    // Python's str.strip() with no argument strips ASCII whitespace.
    std::string inner = match[1].str();
    const size_t first_ch = inner.find_first_not_of(" \t\n\r\f\v");
    if (first_ch == std::string::npos) {
      inner.clear();
    } else {
      inner = inner.substr(first_ch, inner.find_last_not_of(" \t\n\r\f\v") - first_ch + 1);
    }
    // `inner.split(None, 1)` — split on the FIRST whitespace run only.
    const size_t space = inner.find_first_of(" \t\n\r\f\v");
    if (space == std::string::npos) {
      text += inner;
    } else {
      const size_t rest = inner.find_first_not_of(" \t\n\r\f\v", space);
      text += inner.substr(0, space) + " is " +
              (rest == std::string::npos ? std::string() : inner.substr(rest));
    }
    last = static_cast<size_t>(match.position(0) + match.length(0));
  }
  text.append(caption, last, std::string::npos);

  // encoders.py:62-74 — per line: ATX heading, then two bullet forms, then the
  // bold/italic unwraps, then a trailing-whitespace rstrip.
  static const std::regex kHeading(R"(^\s{0,3}#{1,6}\s+)");
  static const std::regex kBulletA(R"(^\s*[*+-]\s+)");
  static const std::regex kBulletB(R"(^\s*\*\s+)");
  static const std::regex kBold(R"(\*\*([^*]+)\*\*)");
  static const std::regex kItalic(R"((^|[^*])\*([^*\n]+)\*($|[^*]))");
  std::vector<std::string> lines;
  {
    std::string line;
    for (const char ch : text) {
      if (ch == '\n') {
        lines.push_back(line);
        line.clear();
      } else {
        line += ch;
      }
    }
    lines.push_back(line);
    // Python's splitlines() drops a single trailing empty field.
    if (!text.empty() && text.back() == '\n') lines.pop_back();
  }
  std::string joined;
  for (size_t i = 0; i < lines.size(); ++i) {
    std::string line = std::regex_replace(lines[i], kHeading, "", std::regex_constants::format_first_only);
    line = std::regex_replace(line, kBulletA, "", std::regex_constants::format_first_only);
    line = std::regex_replace(line, kBulletB, "", std::regex_constants::format_first_only);
    while (line.find("**") != std::string::npos) {
      const std::string updated =
          std::regex_replace(line, kBold, "$1");
      if (updated == line) break;
      line = updated;
    }
    // The lookarounds of `(?<!\*)\*([^*\n]+)\*(?!\*)` are emulated by capturing
    // the neighbours; std::regex has no lookbehind.
    line = std::regex_replace(line, kItalic, "$1$2$3");
    const size_t keep = line.find_last_not_of(" \t\n\r\f\v");
    line = keep == std::string::npos ? std::string() : line.substr(0, keep + 1);
    if (i != 0) joined += '\n';
    joined += line;
  }
  text = joined;

  // encoders.py:75 — a whole-line horizontal rule becomes an EMPTY line (it is
  // not deleted), which is why the blank-run collapse below still has work.
  static const std::regex kRule(R"(^\s*[-*_]{3,}\s*$)");
  {
    std::string rebuilt;
    size_t pos = 0;
    while (pos <= text.size()) {
      const size_t nl = text.find('\n', pos);
      const size_t stop = nl == std::string::npos ? text.size() : nl;
      const std::string line = text.substr(pos, stop - pos);
      rebuilt += std::regex_replace(line, kRule, "");
      if (nl == std::string::npos) break;
      rebuilt += '\n';
      pos = nl + 1;
    }
    text = rebuilt;
  }

  // encoders.py:76-77
  for (const char* needle : {"\xe2\x80\xa2 ", "    "}) {
    const std::string pattern(needle);
    size_t at = 0;
    while ((at = text.find(pattern, at)) != std::string::npos) {
      text.erase(at, pattern.size());
    }
  }
  static const std::regex kBlankRun(R"(\n{2,})");
  return std::regex_replace(text, kBlankRun, "\n");
}

std::string NormalizeLyrics(const std::string& lyrics) {
  // encoders.py:82-86 — a line whose LEADING run is structural tags keeps only
  // those tags; any other line is untouched.
  static const std::regex kLeadingTags(R"(^[ \t]*((?:\[[^\]]+\][ \t]*)+))");
  std::string text;
  size_t pos = 0;
  bool first = true;
  while (true) {
    const size_t nl = lyrics.find('\n', pos);
    const std::string line =
        lyrics.substr(pos, nl == std::string::npos ? std::string::npos : nl - pos);
    std::smatch match;
    std::string out = line;
    if (std::regex_search(line, match, kLeadingTags)) {
      out = match[1].str();
      const size_t keep = out.find_last_not_of(" \t\n\r\f\v");
      const size_t drop = out.find_first_not_of(" \t\n\r\f\v");
      out = keep == std::string::npos ? std::string() : out.substr(drop, keep - drop + 1);
    }
    if (!first) text += '\n';
    text += out;
    first = false;
    if (nl == std::string::npos) break;
    pos = nl + 1;
  }

  // encoders.py:87-89. THE ORDER IS LOAD-BEARING; see the header note.
  const auto replace_all = [](std::string& s, const std::string& from, const std::string& to) {
    size_t at = 0;
    while ((at = s.find(from, at)) != std::string::npos) {
      s.replace(at, from.size(), to);
      at += to.size();
    }
  };
  replace_all(text, "] ", "]\n");
  replace_all(text, " [", "\n[");
  replace_all(text, " ^ ", "\n");

  // encoders.py:90 — lower-case the tag body only.
  std::string lowered;
  for (size_t i = 0; i < text.size();) {
    if (text[i] == '[') {
      const size_t close = text.find(']', i + 1);
      if (close != std::string::npos && close > i + 1) {
        lowered += '[';
        for (size_t j = i + 1; j < close; ++j) {
          lowered += static_cast<char>(
              std::tolower(static_cast<unsigned char>(text[j])));
        }
        lowered += ']';
        i = close + 1;
        continue;
      }
    }
    lowered += text[i++];
  }
  return std::string("[start]\n") + lowered;
}

std::string AssembleArPrompt(const std::string& prompt, const std::string& lyrics) {
  if (prompt.find_first_not_of(" \t\n\r\f\v") == std::string::npos) {
    Fail("MiniMax-Music3: `prompt` (the music description) must be a non-empty string");
  }
  if (lyrics.find_first_not_of(" \t\n\r\f\v") == std::string::npos) {
    Fail("MiniMax-Music3: `lyrics` must be a non-empty string");
  }
  return std::string(kImStart) + kCaptionStart + CleanCaption(prompt) + kCaptionEnd +
         kLyricsStart + NormalizeLyrics(lyrics) + kLyricsEnd + kImEnd + kAudioStart;
}

std::vector<int32_t> UnconditionalPromptIds(const std::vector<int32_t>& ids) {
  if (static_cast<int64_t>(ids.size()) < 4) {
    Fail("MiniMax-Music3: the prompt has " + std::to_string(ids.size()) +
         " tokens; the unconditional rewrite (encoders.py:217) needs at least 4 so "
         "that [1:-2] is a non-empty slice");
  }
  if (static_cast<int64_t>(ids.size()) > kMaxPromptTokens) {
    Fail("MiniMax-Music3: the assembled prompt has " + std::to_string(ids.size()) +
         " tokens; the maximum is " + std::to_string(kMaxPromptTokens));
  }
  std::vector<int32_t> out = ids;
  for (size_t i = 1; i + 2 < out.size(); ++i) out[i] = kAudioCfgTokenId;
  return out;
}

int64_t MaxArFrames(double audio_duration_s, double frame_rate) {
  if (!(audio_duration_s > 0.0)) {
    Fail("MiniMax-Music3: `audio_duration` must be positive, got " +
         std::to_string(audio_duration_s));
  }
  const int64_t frames = static_cast<int64_t>(audio_duration_s * frame_rate);
  if (frames == 0) {
    Fail("MiniMax-Music3: `audio_duration` " + std::to_string(audio_duration_s) +
         " is shorter than one audio frame (1 / " + std::to_string(frame_rate) + " s)");
  }
  return std::min<int64_t>(frames, kMaxAudioFrames);
}

// ---------------------------------------------------------------------------
// Logit pipeline
// ---------------------------------------------------------------------------

std::vector<bool> SemanticVocabMask(int64_t vocab_size, int64_t code_offset,
                                    int64_t semantic_vocab_size, int32_t end_token_id) {
  if (vocab_size <= 0) Fail("MiniMax-Music3: vocab_size must be positive");
  if (code_offset < 0 || code_offset + semantic_vocab_size > vocab_size) {
    Fail("MiniMax-Music3: the semantic code window [" + std::to_string(code_offset) + ", " +
         std::to_string(code_offset + semantic_vocab_size) + ") does not fit in a vocabulary of " +
         std::to_string(vocab_size));
  }
  if (end_token_id < 0 || end_token_id >= vocab_size) {
    Fail("MiniMax-Music3: the audio-end token " + std::to_string(end_token_id) +
         " is outside a vocabulary of " + std::to_string(vocab_size));
  }
  std::vector<bool> blocked(static_cast<size_t>(vocab_size), true);
  for (int64_t i = code_offset; i < code_offset + semantic_vocab_size; ++i) {
    blocked[static_cast<size_t>(i)] = false;
  }
  blocked[static_cast<size_t>(end_token_id)] = false;
  return blocked;
}

std::vector<float> GuidedSemanticLogits(const std::vector<float>& conditional,
                                        const std::vector<float>& unconditional,
                                        const std::vector<bool>& blocked, int64_t cfg_top_k,
                                        double cfg_scale) {
  const size_t n = conditional.size();
  if (unconditional.size() != n || blocked.size() != n) {
    Fail("MiniMax-Music3: guided-logit inputs disagree on vocabulary size (" +
         std::to_string(conditional.size()) + ", " + std::to_string(unconditional.size()) +
         ", " + std::to_string(blocked.size()) + ")");
  }
  if (cfg_top_k <= 0 || cfg_top_k > static_cast<int64_t>(n)) {
    Fail("MiniMax-Music3: cfg_top_k " + std::to_string(cfg_top_k) +
         " is out of range for a vocabulary of " + std::to_string(n));
  }
  const float kNegInf = -std::numeric_limits<float>::infinity();
  // encoders.py:326 — the mask is applied to the RAW rows, before guidance.
  std::vector<float> cond(n), uncond(n);
  for (size_t i = 0; i < n; ++i) {
    cond[i] = blocked[i] ? kNegInf : conditional[i];
    uncond[i] = blocked[i] ? kNegInf : unconditional[i];
  }
  // encoders.py:328
  std::vector<float> guided(n);
  for (size_t i = 0; i < n; ++i) {
    guided[i] = static_cast<float>(static_cast<double>(uncond[i]) +
                                   (static_cast<double>(cond[i]) - static_cast<double>(uncond[i])) *
                                       cfg_scale);
  }
  // encoders.py:331-332 — the threshold is the CONDITIONAL row's k-th largest.
  std::vector<size_t> order(n);
  std::iota(order.begin(), order.end(), size_t{0});
  std::partial_sort(order.begin(), order.begin() + static_cast<size_t>(cfg_top_k), order.end(),
                    [&cond](size_t a, size_t b) { return cond[a] > cond[b]; });
  const float threshold = cond[order[static_cast<size_t>(cfg_top_k) - 1]];
  for (size_t i = 0; i < n; ++i) {
    if (cond[i] < threshold) guided[i] = kNegInf;
  }
  // encoders.py:333 — the re-mask that keeps a NaN from becoming a candidate.
  for (size_t i = 0; i < n; ++i) {
    if (blocked[i]) guided[i] = kNegInf;
  }
  return guided;
}

std::vector<float> GuidedDepthLogits(const std::vector<float>& conditional,
                                     const std::vector<float>& unconditional,
                                     double cfg_scale) {
  if (conditional.size() != unconditional.size()) {
    Fail("MiniMax-Music3: depth CFG rows disagree on size (" +
         std::to_string(conditional.size()) + " vs " + std::to_string(unconditional.size()) + ")");
  }
  std::vector<float> guided(conditional.size());
  for (size_t i = 0; i < conditional.size(); ++i) {
    const double c = conditional[i];
    const double u = unconditional[i];
    guided[i] = static_cast<float>(u + (c - u) * cfg_scale);
  }
  return guided;
}

std::vector<float> TopKProbabilities(const std::vector<float>& logits, int64_t top_k) {
  const size_t n = logits.size();
  if (n == 0) Fail("MiniMax-Music3: TopKProbabilities needs a non-empty row");
  if (top_k <= 0) Fail("MiniMax-Music3: top_k must be positive");
  const float kNegInf = -std::numeric_limits<float>::infinity();
  // encoders.py:95 — nan_to_num BEFORE anything else, with upstream's finite
  // substitutes; a -inf becomes -1e9, which is why a masked position can still
  // be selected when it survives the top-k.
  std::vector<float> values(n);
  for (size_t i = 0; i < n; ++i) {
    const float v = logits[i];
    values[i] = std::isnan(v) ? -1e9f : (v == std::numeric_limits<float>::infinity()
                                             ? 1e9f
                                             : (v == kNegInf ? -1e9f : v));
  }
  const size_t k = std::min<size_t>(static_cast<size_t>(top_k), n);
  std::vector<size_t> order(n);
  std::iota(order.begin(), order.end(), size_t{0});
  std::partial_sort(order.begin(), order.begin() + k, order.end(),
                    [&values](size_t a, size_t b) { return values[a] > values[b]; });
  const float threshold = values[order[k - 1]];
  for (size_t i = 0; i < n; ++i) {
    if (values[i] < threshold) values[i] = kNegInf;
  }
  float max_value = kNegInf;
  for (const float v : values) max_value = std::max(max_value, v);
  double sum = 0.0;
  std::vector<double> probs(n);
  for (size_t i = 0; i < n; ++i) {
    probs[i] = values[i] == kNegInf ? 0.0 : std::exp(static_cast<double>(values[i] - max_value));
    sum += probs[i];
  }
  // encoders.py:99-100 — nan_to_num on the softmax, then renormalize with a
  // 1e-12 floor on the denominator.
  const double denom = std::max(sum, 1e-12);
  std::vector<float> out(n);
  for (size_t i = 0; i < n; ++i) {
    const double p = std::isnan(probs[i]) ? 0.0 : probs[i];
    out[i] = static_cast<float>(p / denom);
  }
  return out;
}

// ---------------------------------------------------------------------------
// Condition mix
// ---------------------------------------------------------------------------

int64_t ConditionLatentLength(int64_t num_frames, const ConditionMixConfig& config) {
  if (num_frames <= 0) Fail("MiniMax-Music3: the condition mix needs at least one frame");
  if (config.input_sampling_rate <= 0 || config.output_hop_length <= 0) {
    Fail("MiniMax-Music3: the condition encoder's rates must be positive");
  }
  const double scaled = static_cast<double>(num_frames) *
                        static_cast<double>(config.output_sampling_rate) /
                        static_cast<double>(config.input_sampling_rate) *
                        static_cast<double>(config.input_hop_length) /
                        static_cast<double>(config.output_hop_length);
  return std::max<int64_t>(1, static_cast<int64_t>(scaled));
}

std::vector<float> ConditionLayerWeights(const std::vector<float>& layer_weight_logits,
                                         ArCompute compute) {
  if (layer_weight_logits.empty()) {
    Fail("MiniMax-Music3: the condition mix has no layer logits");
  }
  const float max_logit =
      *std::max_element(layer_weight_logits.begin(), layer_weight_logits.end());
  double sum = 0.0;
  std::vector<double> exps(layer_weight_logits.size());
  for (size_t i = 0; i < layer_weight_logits.size(); ++i) {
    exps[i] = std::exp(static_cast<double>(layer_weight_logits[i] - max_logit));
    sum += exps[i];
  }
  std::vector<float> out(layer_weight_logits.size());
  for (size_t i = 0; i < out.size(); ++i) out[i] = Store(exps[i] / sum, compute);
  return out;
}

std::vector<float> NearestInterpolate1d(const std::vector<float>& in, int64_t channels,
                                        int64_t in_len, int64_t out_len) {
  if (in_len <= 0 || out_len <= 0 || channels <= 0) {
    Fail("MiniMax-Music3: NearestInterpolate1d needs positive extents");
  }
  if (static_cast<int64_t>(in.size()) != channels * in_len) {
    Fail("MiniMax-Music3: NearestInterpolate1d input is " + std::to_string(in.size()) +
         " values, expected " + std::to_string(channels * in_len));
  }
  const double scale = static_cast<double>(in_len) / static_cast<double>(out_len);
  std::vector<float> out(static_cast<size_t>(channels * out_len));
  for (int64_t t = 0; t < out_len; ++t) {
    const int64_t src = std::min<int64_t>(
        in_len - 1, static_cast<int64_t>(std::floor(static_cast<double>(t) * scale)));
    for (int64_t c = 0; c < channels; ++c) {
      out[static_cast<size_t>(c * out_len + t)] = in[static_cast<size_t>(c * in_len + src)];
    }
  }
  return out;
}

std::vector<float> ConditionMix(const std::vector<float>& hidden_states, int64_t num_frames,
                                const ConditionMixConfig& config,
                                const ConditionMixWeights& weights, ArCompute compute) {
  const int64_t layers = config.num_condition_layers;
  const int64_t hidden = config.condition_hidden_dim;
  const int64_t out_dim = config.out_dim;
  if (static_cast<int64_t>(hidden_states.size()) != num_frames * layers * hidden) {
    Fail("MiniMax-Music3: the condition mix got " + std::to_string(hidden_states.size()) +
         " hidden values, expected frames*layers*hidden = " +
         std::to_string(num_frames * layers * hidden));
  }
  if (static_cast<int64_t>(weights.layer_weight_logits.size()) != layers) {
    Fail("MiniMax-Music3: layer_weight_logits has " +
         std::to_string(weights.layer_weight_logits.size()) + " entries, expected " +
         std::to_string(layers));
  }
  if (weights.layer_scale.size() != 1) {
    Fail("MiniMax-Music3: layer_scale must be the single learned scalar, got " +
         std::to_string(weights.layer_scale.size()) + " values");
  }
  if (static_cast<int64_t>(weights.proj_weight.size()) != out_dim * hidden * 3) {
    Fail("MiniMax-Music3: proj.weight has " + std::to_string(weights.proj_weight.size()) +
         " values, expected out_dim*hidden*3 = " + std::to_string(out_dim * hidden * 3));
  }
  if (static_cast<int64_t>(weights.proj_bias.size()) != out_dim) {
    Fail("MiniMax-Music3: proj.bias has " + std::to_string(weights.proj_bias.size()) +
         " values, expected " + std::to_string(out_dim));
  }

  const std::vector<float> layer_weights =
      ConditionLayerWeights(weights.layer_weight_logits, compute);
  const double scale = static_cast<double>(weights.layer_scale[0]);

  // :59-63 — the einsum "blht,l->bht" over a LAYER-MAJOR last axis, then the one
  // scalar. Emitted as [hidden, frames] because the Conv1d that follows is over
  // time.
  std::vector<float> mixed(static_cast<size_t>(hidden * num_frames));
  for (int64_t h = 0; h < hidden; ++h) {
    for (int64_t t = 0; t < num_frames; ++t) {
      double acc = 0.0;
      for (int64_t l = 0; l < layers; ++l) {
        acc += static_cast<double>(hidden_states[static_cast<size_t>(
                   t * layers * hidden + l * hidden + h)]) *
               static_cast<double>(layer_weights[static_cast<size_t>(l)]);
      }
      // :62 einsum then :63 the scalar — two ops, so two stores.
      mixed[static_cast<size_t>(h * num_frames + t)] =
          Store(static_cast<double>(Store(acc, compute)) * scale, compute);
    }
  }

  // :46,:64 — nn.Conv1d(hidden, out_dim, kernel_size=3, padding=1) through the
  // shared 1-D primitives, so this port has no second convolution of its own.
  int64_t padded_len = 0;
  const std::vector<float> padded =
      vocoder1d::Pad1d(mixed, hidden, num_frames, 1, 1, /*replicate=*/false, &padded_len);
  int64_t conv_len = 0;
  std::vector<float> projected =
      vocoder1d::Conv1d(padded, hidden, padded_len, weights.proj_weight, &weights.proj_bias,
                        out_dim, /*kernel=*/3, /*stride=*/1, /*dilation=*/1, /*groups=*/1,
                        &conv_len);
  for (float& value : projected) value = Store(value, compute);
  if (conv_len != num_frames) {
    Fail("MiniMax-Music3: the condition projection produced " + std::to_string(conv_len) +
         " frames, expected " + std::to_string(num_frames));
  }

  // :65-76 — nearest resample onto the latent timeline, then transpose(1, 2) so
  // the return is [latent_length, out_dim].
  const int64_t latent_length = ConditionLatentLength(num_frames, config);
  const std::vector<float> resampled =
      NearestInterpolate1d(projected, out_dim, conv_len, latent_length);
  std::vector<float> out(static_cast<size_t>(latent_length * out_dim));
  for (int64_t t = 0; t < latent_length; ++t) {
    for (int64_t c = 0; c < out_dim; ++c) {
      out[static_cast<size_t>(t * out_dim + c)] =
          resampled[static_cast<size_t>(c * latent_length + t)];
    }
  }
  return out;
}

// ---------------------------------------------------------------------------
// Depth decoder
// ---------------------------------------------------------------------------

std::vector<float> RmsNorm(const std::vector<float>& x, int64_t rows, int64_t dim,
                           const std::vector<float>& weight, double eps, ArCompute compute) {
  if (static_cast<int64_t>(x.size()) != rows * dim) {
    Fail("MiniMax-Music3: RmsNorm got " + std::to_string(x.size()) + " values, expected " +
         std::to_string(rows * dim));
  }
  if (static_cast<int64_t>(weight.size()) != dim) {
    Fail("MiniMax-Music3: RmsNorm weight has " + std::to_string(weight.size()) +
         " values, expected " + std::to_string(dim));
  }
  std::vector<float> out(x.size());
  for (int64_t r = 0; r < rows; ++r) {
    double sum = 0.0;
    for (int64_t c = 0; c < dim; ++c) {
      const double v = x[static_cast<size_t>(r * dim + c)];
      sum += v * v;
    }
    // normalization.py:600 computes the variance on the FLOAT32 upcast, so the
    // reduction is wide at both dtypes.
    const double inv = 1.0 / std::sqrt(static_cast<double>(static_cast<float>(
                                           sum / static_cast<double>(dim))) +
                                       eps);
    for (int64_t c = 0; c < dim; ++c) {
      // :601 promotes to float32, :605 casts BACK to the weight dtype, and only
      // :606 applies the affine weight. Two stores, in that order.
      const double normed = static_cast<double>(x[static_cast<size_t>(r * dim + c)]) * inv;
      out[static_cast<size_t>(r * dim + c)] =
          Store(static_cast<double>(Store(normed, compute)) *
                    static_cast<double>(weight[static_cast<size_t>(c)]),
                compute);
    }
  }
  return out;
}

std::vector<float> LinearNoBias(const std::vector<float>& x, int64_t rows, int64_t in_dim,
                                const std::vector<float>& weight, int64_t out_dim,
                                ArCompute compute) {
  if (static_cast<int64_t>(x.size()) != rows * in_dim) {
    Fail("MiniMax-Music3: LinearNoBias input is " + std::to_string(x.size()) +
         " values, expected " + std::to_string(rows * in_dim));
  }
  if (static_cast<int64_t>(weight.size()) != out_dim * in_dim) {
    Fail("MiniMax-Music3: LinearNoBias weight is " + std::to_string(weight.size()) +
         " values, expected out_dim*in_dim = " + std::to_string(out_dim * in_dim));
  }
  std::vector<float> out(static_cast<size_t>(rows * out_dim));
  for (int64_t r = 0; r < rows; ++r) {
    for (int64_t o = 0; o < out_dim; ++o) {
      double acc = 0.0;
      const float* xr = x.data() + r * in_dim;
      const float* wo = weight.data() + o * in_dim;
      for (int64_t i = 0; i < in_dim; ++i) acc += static_cast<double>(xr[i]) * wo[i];
      out[static_cast<size_t>(r * out_dim + o)] = Store(acc, compute);
    }
  }
  return out;
}

namespace {

// Causal scaled-dot-product attention over [seq, heads*head_dim] q/k/v laid out
// as torch's `view(batch, seq, heads, head_dim)` — head is the FAST axis inside
// a row (minimax_music3_rvq_depth_decoder.py:39-41).
std::vector<float> CausalAttention(const std::vector<float>& q, const std::vector<float>& k,
                                   const std::vector<float>& v, int64_t seq, int64_t heads,
                                   int64_t head_dim, ArCompute compute) {
  const double inv_sqrt = 1.0 / std::sqrt(static_cast<double>(head_dim));
  std::vector<float> out(static_cast<size_t>(seq * heads * head_dim), 0.0f);
  std::vector<double> scores(static_cast<size_t>(seq));
  for (int64_t h = 0; h < heads; ++h) {
    for (int64_t i = 0; i < seq; ++i) {
      double max_score = -std::numeric_limits<double>::infinity();
      for (int64_t j = 0; j <= i; ++j) {
        double acc = 0.0;
        for (int64_t d = 0; d < head_dim; ++d) {
          acc += static_cast<double>(q[static_cast<size_t>((i * heads + h) * head_dim + d)]) *
                 static_cast<double>(k[static_cast<size_t>((j * heads + h) * head_dim + d)]);
        }
        scores[static_cast<size_t>(j)] = acc * inv_sqrt;
        max_score = std::max(max_score, scores[static_cast<size_t>(j)]);
      }
      double sum = 0.0;
      for (int64_t j = 0; j <= i; ++j) {
        scores[static_cast<size_t>(j)] = std::exp(scores[static_cast<size_t>(j)] - max_score);
        sum += scores[static_cast<size_t>(j)];
      }
      for (int64_t d = 0; d < head_dim; ++d) {
        double acc = 0.0;
        for (int64_t j = 0; j <= i; ++j) {
          acc += scores[static_cast<size_t>(j)] *
                 static_cast<double>(v[static_cast<size_t>((j * heads + h) * head_dim + d)]);
        }
        out[static_cast<size_t>((i * heads + h) * head_dim + d)] = Store(acc / sum, compute);
      }
    }
  }
  return out;
}

}  // namespace

std::vector<float> DepthDecoderForward(const std::vector<float>& inputs_embeds, int64_t seq_len,
                                       const DepthDecoderConfig& config,
                                       const DepthDecoderWeights& weights, ArCompute compute) {
  const int64_t hidden = config.hidden_size;
  if (seq_len <= 0) Fail("MiniMax-Music3: the depth decoder needs at least one position");
  if (seq_len > config.max_position_embeddings) {
    Fail("MiniMax-Music3: the depth decoder was given " + std::to_string(seq_len) +
         " positions but max_position_embeddings is " +
         std::to_string(config.max_position_embeddings) +
         "; pos_embedding has no row for the rest");
  }
  if (static_cast<int64_t>(inputs_embeds.size()) != seq_len * hidden) {
    Fail("MiniMax-Music3: the depth decoder got " + std::to_string(inputs_embeds.size()) +
         " input values, expected seq*hidden = " + std::to_string(seq_len * hidden));
  }
  if (static_cast<int64_t>(weights.layers.size()) != config.num_layers) {
    Fail("MiniMax-Music3: the depth decoder has " + std::to_string(weights.layers.size()) +
         " layer weight sets, expected " + std::to_string(config.num_layers));
  }
  if (static_cast<int64_t>(weights.pos_embedding.size()) !=
      config.max_position_embeddings * hidden) {
    Fail("MiniMax-Music3: pos_embedding is " + std::to_string(weights.pos_embedding.size()) +
         " values, expected " + std::to_string(config.max_position_embeddings * hidden));
  }
  const int64_t heads = config.num_attention_heads;
  const int64_t head_dim = config.head_dim();
  if (heads * head_dim != hidden) {
    Fail("MiniMax-Music3: hidden_size " + std::to_string(hidden) +
         " is not divisible by num_attention_heads " + std::to_string(heads));
  }

  // :138-139 — positions are arange(seq) into a LEARNED table; there is no RoPE.
  std::vector<float> hidden_states(inputs_embeds.size());
  for (int64_t t = 0; t < seq_len; ++t) {
    for (int64_t c = 0; c < hidden; ++c) {
      hidden_states[static_cast<size_t>(t * hidden + c)] =
          Store(static_cast<double>(inputs_embeds[static_cast<size_t>(t * hidden + c)]) +
                    weights.pos_embedding[static_cast<size_t>(t * hidden + c)],
                compute);
    }
  }

  for (const DepthDecoderLayerWeights& layer : weights.layers) {
    // :86 — pre-norm attention with a residual add.
    const std::vector<float> normed =
        RmsNorm(hidden_states, seq_len, hidden, layer.input_layernorm, 1e-6, compute);
    const std::vector<float> q =
        LinearNoBias(normed, seq_len, hidden, layer.to_q, hidden, compute);
    const std::vector<float> k =
        LinearNoBias(normed, seq_len, hidden, layer.to_k, hidden, compute);
    const std::vector<float> v =
        LinearNoBias(normed, seq_len, hidden, layer.to_v, hidden, compute);
    const std::vector<float> attended =
        CausalAttention(q, k, v, seq_len, heads, head_dim, compute);
    const std::vector<float> projected =
        LinearNoBias(attended, seq_len, hidden, layer.to_out, hidden, compute);
    for (size_t i = 0; i < hidden_states.size(); ++i) {
      hidden_states[i] = Store(static_cast<double>(hidden_states[i]) + projected[i], compute);
    }

    // :87-88 — SwiGLU MLP with a residual add.
    const std::vector<float> post =
        RmsNorm(hidden_states, seq_len, hidden, layer.post_attention_layernorm, 1e-6, compute);
    const std::vector<float> gate =
        LinearNoBias(post, seq_len, hidden, layer.gate_proj, config.intermediate_size, compute);
    const std::vector<float> up =
        LinearNoBias(post, seq_len, hidden, layer.up_proj, config.intermediate_size, compute);
    std::vector<float> activated(gate.size());
    for (size_t i = 0; i < gate.size(); ++i) {
      // F.silu then the elementwise product: two ops, two stores.
      const double g = gate[i];
      const double silu = Store(g / (1.0 + std::exp(-g)), compute);
      activated[i] = Store(silu * static_cast<double>(up[i]), compute);
    }
    const std::vector<float> down = LinearNoBias(activated, seq_len, config.intermediate_size,
                                                 layer.down_proj, hidden, compute);
    for (size_t i = 0; i < hidden_states.size(); ++i) {
      hidden_states[i] = Store(static_cast<double>(hidden_states[i]) + down[i], compute);
    }
  }

  // :142
  return RmsNorm(hidden_states, seq_len, hidden, weights.norm, 1e-6, compute);
}

std::vector<float> DepthSequenceEmbeds(const std::vector<float>& last_hidden,
                                       const std::vector<float>& semantic_embed,
                                       const std::vector<int32_t>& residual_codes,
                                       const DepthDecoderConfig& config,
                                       const DepthDecoderWeights& weights, ArCompute compute) {
  const int64_t hidden = config.hidden_size;
  if (static_cast<int64_t>(last_hidden.size()) != hidden) {
    Fail("MiniMax-Music3: last_hidden is " + std::to_string(last_hidden.size()) +
         " values, expected hidden_size = " + std::to_string(hidden));
  }
  if (static_cast<int64_t>(semantic_embed.size()) != hidden) {
    Fail("MiniMax-Music3: the semantic embedding row is " +
         std::to_string(semantic_embed.size()) + " values, expected hidden_size = " +
         std::to_string(hidden));
  }
  if (static_cast<int64_t>(residual_codes.size()) > config.residual_codebooks() - 1) {
    Fail("MiniMax-Music3: the depth sequence carries at most " +
         std::to_string(config.residual_codebooks() - 1) +
         " residual codes (the last codebook is predicted, never fed back), got " +
         std::to_string(residual_codes.size()));
  }
  std::vector<float> rows;
  rows.reserve(static_cast<size_t>((2 + residual_codes.size()) * hidden));
  rows.insert(rows.end(), last_hidden.begin(), last_hidden.end());
  rows.insert(rows.end(), semantic_embed.begin(), semantic_embed.end());
  for (size_t index = 0; index < residual_codes.size(); ++index) {
    const int32_t code = residual_codes[index];
    if (code < 0 || code >= config.audio_vocab_size) {
      Fail("MiniMax-Music3: residual code " + std::to_string(code) + " at depth step " +
           std::to_string(index + 1) + " is outside [0, " +
           std::to_string(config.audio_vocab_size) + ")");
    }
    // encoders.py:140 — `index` there is ONE-based, so the offset is (index-1).
    const int64_t row = static_cast<int64_t>(index) * config.audio_vocab_size + code;
    const size_t at = static_cast<size_t>(row * hidden);
    if (at + static_cast<size_t>(hidden) > weights.audio_embeddings.size()) {
      Fail("MiniMax-Music3: audio_embeddings row " + std::to_string(row) +
           " is past the end of a table of " +
           std::to_string(weights.audio_embeddings.size() / static_cast<size_t>(hidden)) +
           " rows");
    }
    rows.insert(rows.end(), weights.audio_embeddings.begin() + static_cast<int64_t>(at),
                weights.audio_embeddings.begin() + static_cast<int64_t>(at) + hidden);
  }
  const int64_t seq = 2 + static_cast<int64_t>(residual_codes.size());
  return LinearNoBias(rows, seq, hidden, weights.projection, hidden, compute);
}

std::vector<float> AudioHeadLogits(const std::vector<float>& hidden, int64_t head_index,
                                   const DepthDecoderConfig& config,
                                   const DepthDecoderWeights& weights, ArCompute compute) {
  if (head_index < 0 || head_index >= static_cast<int64_t>(weights.audio_heads.size())) {
    Fail("MiniMax-Music3: audio head " + std::to_string(head_index) + " is outside [0, " +
         std::to_string(weights.audio_heads.size()) + ")");
  }
  return LinearNoBias(hidden, 1, config.hidden_size,
                      weights.audio_heads[static_cast<size_t>(head_index)],
                      config.audio_vocab_size, compute);
}

std::vector<float> FrameHiddenRow(const std::vector<float>& last_hidden,
                                  const std::vector<float>& depth_hidden_states, int64_t seq_len,
                                  const DepthDecoderConfig& config) {
  const int64_t hidden = config.hidden_size;
  if (static_cast<int64_t>(last_hidden.size()) != hidden) {
    Fail("MiniMax-Music3: last_hidden is " + std::to_string(last_hidden.size()) +
         " values, expected " + std::to_string(hidden));
  }
  if (static_cast<int64_t>(depth_hidden_states.size()) != seq_len * hidden) {
    Fail("MiniMax-Music3: the depth hidden block is " +
         std::to_string(depth_hidden_states.size()) + " values, expected " +
         std::to_string(seq_len * hidden));
  }
  // encoders.py:131-132 — step index i takes the LAST position of a sequence of
  // length i+1, so over one whole-sequence forward that is positions 1..seq-1.
  if (seq_len != config.num_codebooks) {
    Fail("MiniMax-Music3: a frame row needs a depth sequence of exactly num_codebooks = " +
         std::to_string(config.num_codebooks) + " positions, got " + std::to_string(seq_len));
  }
  std::vector<float> row;
  row.reserve(static_cast<size_t>(config.num_codebooks * hidden));
  row.insert(row.end(), last_hidden.begin(), last_hidden.end());
  row.insert(row.end(), depth_hidden_states.begin() + hidden, depth_hidden_states.end());
  return row;
}

std::vector<float> EmbedAudioFrame(const std::vector<float>& lm_semantic_embed,
                                   const std::vector<int32_t>& residual_codes,
                                   const DepthDecoderConfig& config,
                                   const DepthDecoderWeights& weights, ArCompute compute) {
  const int64_t hidden = config.hidden_size;
  if (static_cast<int64_t>(lm_semantic_embed.size()) != hidden) {
    Fail("MiniMax-Music3: the language model's semantic row is " +
         std::to_string(lm_semantic_embed.size()) + " values, expected " + std::to_string(hidden));
  }
  if (static_cast<int64_t>(residual_codes.size()) != config.residual_codebooks()) {
    Fail("MiniMax-Music3: the frame feedback needs all " +
         std::to_string(config.residual_codebooks()) + " residual codes, got " +
         std::to_string(residual_codes.size()));
  }
  std::vector<double> acc(static_cast<size_t>(hidden));
  for (int64_t c = 0; c < hidden; ++c) acc[static_cast<size_t>(c)] = lm_semantic_embed[static_cast<size_t>(c)];
  for (size_t j = 0; j < residual_codes.size(); ++j) {
    const int32_t code = residual_codes[j];
    if (code < 0 || code >= config.audio_vocab_size) {
      Fail("MiniMax-Music3: residual code " + std::to_string(code) + " for codebook " +
           std::to_string(j + 1) + " is outside [0, " +
           std::to_string(config.audio_vocab_size) + ")");
    }
    // encoders.py:110-113 — `arange(num_codebooks-1) * audio_vocab_size` is a
    // ZERO-based offset here.
    const int64_t row = static_cast<int64_t>(j) * config.audio_vocab_size + code;
    const size_t at = static_cast<size_t>(row * hidden);
    for (int64_t c = 0; c < hidden; ++c) {
      acc[static_cast<size_t>(c)] += weights.audio_embeddings[at + static_cast<size_t>(c)];
    }
  }
  const double scale = 1.0 / std::sqrt(static_cast<double>(config.num_codebooks));
  std::vector<float> out(static_cast<size_t>(hidden));
  for (int64_t c = 0; c < hidden; ++c) {
    // encoders.py:113-115 — the residual sum, the add, then the scale.
    out[static_cast<size_t>(c)] =
        Store(static_cast<double>(Store(acc[static_cast<size_t>(c)], compute)) * scale, compute);
  }
  return out;
}

}  // namespace music3
}  // namespace models
}  // namespace vllm
