// IndexTTS-2.5 talker embedding scaffolding. See talker.h.
#include "vllm/model_executor/models/talker.h"

#include <cstddef>
#include <vector>

#include "vt/dtype.h"

namespace vllm {
namespace models {
namespace talker {

std::vector<float> PositionRows(const std::vector<float>& table, int64_t dim, int64_t seq_len) {
  VT_CHECK(dim > 0 && seq_len >= 0, "talker: bad dims");
  VT_CHECK(table.size() >= static_cast<size_t>(seq_len * dim),
           "talker: position table shorter than the sequence");
  std::vector<float> out(static_cast<size_t>(seq_len * dim));
  for (int64_t t = 0; t < seq_len; ++t) {
    for (int64_t d = 0; d < dim; ++d) {
      out[static_cast<size_t>(t * dim + d)] = table[static_cast<size_t>(t * dim + d)];
    }
  }
  return out;
}

std::vector<float> PositionRowAt(const std::vector<float>& table, int64_t dim, int64_t index) {
  VT_CHECK(index >= 0, "talker: position index must be non-negative");
  VT_CHECK(table.size() >= static_cast<size_t>((index + 1) * dim),
           "talker: position index past the end of the table");
  std::vector<float> out(static_cast<size_t>(dim));
  for (int64_t d = 0; d < dim; ++d) {
    out[static_cast<size_t>(d)] = table[static_cast<size_t>(index * dim + d)];
  }
  return out;
}

std::vector<float> EmbedWithPositions(const std::vector<int64_t>& tokens,
                                      const std::vector<float>& token_table,
                                      const std::vector<float>& pos_table, int64_t dim,
                                      int64_t vocab_size) {
  const int64_t seq_len = static_cast<int64_t>(tokens.size());
  VT_CHECK(token_table.size() == static_cast<size_t>(vocab_size * dim),
           "talker: token table shape");
  std::vector<float> out(static_cast<size_t>(seq_len * dim));
  for (int64_t t = 0; t < seq_len; ++t) {
    const int64_t id = tokens[static_cast<size_t>(t)];
    VT_CHECK(id >= 0 && id < vocab_size, "talker: token id out of range");
    for (int64_t d = 0; d < dim; ++d) {
      out[static_cast<size_t>(t * dim + d)] =
          token_table[static_cast<size_t>(id * dim + d)] +
          pos_table[static_cast<size_t>(t * dim + d)];
    }
  }
  return out;
}

Prompt PrepareInputs(const PromptConfig& cfg, const PromptWeights& w,
                     const std::vector<float>& conditioning, int64_t cond_rows,
                     const std::vector<int64_t>& text_tokens, int64_t lang) {
  const int64_t dim = cfg.dim;
  VT_CHECK(dim > 0 && cond_rows > 0, "talker: dim and cond_rows must be positive");
  VT_CHECK(conditioning.size() == static_cast<size_t>(cond_rows * dim),
           "talker: conditioning must be [cond_rows, dim]");
  VT_CHECK(cfg.text_slots > 0, "talker: text_slots must be positive");

  // Strip any delimiters already present, then re-add exactly one of each.
  std::vector<int64_t> text;
  text.push_back(cfg.start_text_token);
  for (const int64_t tok : text_tokens) {
    if (tok != cfg.start_text_token && tok != cfg.stop_text_token) {
      text.push_back(tok);
    }
  }
  text.push_back(cfg.stop_text_token);

  const int64_t n = static_cast<int64_t>(text.size());
  const int64_t target_len = cond_rows + cfg.text_slots + 2;
  const int64_t padding = cfg.text_slots + 2 - n;
  VT_CHECK(padding >= 0,
           "talker: the text is longer than the declared slots allow");
  VT_CHECK(static_cast<int64_t>(w.text_pos_embedding.size()) >= n * dim,
           "talker: the position table is shorter than this text");

  Prompt out;
  out.target_len = target_len;
  out.embeds.assign(static_cast<size_t>(target_len * dim), 0.0F);

  // [pad][conditioning][text] -- the pad goes FIRST, ahead of the conditioning.
  for (int64_t r = 0; r < cond_rows; ++r) {
    for (int64_t d = 0; d < dim; ++d) {
      out.embeds[static_cast<size_t>((padding + r) * dim + d)] =
          conditioning[static_cast<size_t>(r * dim + d)];
    }
  }
  for (int64_t i = 0; i < n; ++i) {
    const int64_t row = padding + cond_rows + i;
    for (int64_t d = 0; d < dim; ++d) {
      const size_t tok = static_cast<size_t>(text[static_cast<size_t>(i)] * dim + d);
      VT_CHECK(tok < w.text_embedding.size(), "talker: text token out of range");
      out.embeds[static_cast<size_t>(row * dim + d)] =
          w.text_embedding[tok] +
          w.text_pos_embedding[static_cast<size_t>(i * dim + d)] +
          // The language vector reaches EVERY text position.
          w.lang_embedding[static_cast<size_t>(lang * dim + d)];
    }
  }

  // The mask is one longer than the embeddings: the extra slot is the
  // start_mel_token, which is an ID rather than an embedded row. Only the pad
  // is masked out.
  out.attention_mask.assign(static_cast<size_t>(target_len + 1), 1);
  for (int64_t i = 0; i < padding; ++i) {
    out.attention_mask[static_cast<size_t>(i)] = 0;
  }
  out.input_ids.assign(static_cast<size_t>(target_len + 1), 1);
  out.input_ids.back() = cfg.start_mel_token;
  return out;
}

std::vector<int64_t> GenerateMelCodes(const GenerateConfig& cfg, const GenerateWeights& w,
                                      const gpt2::Params& params,
                                      const gpt2::Weights& backbone,
                                      const std::vector<float>& prompt_embeds,
                                      int64_t prompt_len) {
  const int64_t dim = cfg.dim;
  VT_CHECK(dim > 0 && cfg.mel_codes > 0 && cfg.max_mel_tokens > 0,
           "talker: dim, mel_codes and max_mel_tokens must be positive");
  VT_CHECK(prompt_embeds.size() == static_cast<size_t>(prompt_len * dim),
           "talker: prompt_embeds must be [prompt_len, dim]");
  VT_CHECK(params.hidden_size == dim,
           "talker: the backbone hidden size must match the talker dim");
  // Validate the TABLES, not just the ids. Checking a code against `mel_codes`
  // says nothing about whether the embedding table actually has that row, and a
  // short table is then an out-of-bounds READ rather than a refusal -- silent,
  // and it produces plausible logits. The gate caught exactly this.
  VT_CHECK(w.mel_embedding.size() >= static_cast<size_t>(cfg.mel_codes * dim),
           "talker: the mel embedding table is shorter than mel_codes");
  VT_CHECK(w.mel_head_w.size() >= static_cast<size_t>(cfg.mel_codes * dim),
           "talker: the mel head does not cover every mel code");
  VT_CHECK(w.final_norm_w.size() >= static_cast<size_t>(dim) &&
               w.final_norm_b.size() >= static_cast<size_t>(dim),
           "talker: final_norm must have one value per dimension");

  // The prompt, then the start-of-mel row, then whatever we emit. Step 0 is the
  // start token, so its position row is index 0 of the MEL table.
  std::vector<float> seq = prompt_embeds;
  auto append_mel = [&](int64_t code, int64_t step) {
    VT_CHECK(code >= 0 && code < cfg.mel_codes, "talker: mel code out of range");
    VT_CHECK(static_cast<int64_t>(w.mel_pos_embedding.size()) >= (step + 1) * dim,
             "talker: the mel position table is shorter than this generation");
    for (int64_t d = 0; d < dim; ++d) {
      seq.push_back(w.mel_embedding[static_cast<size_t>(code * dim + d)] +
                    w.mel_pos_embedding[static_cast<size_t>(step * dim + d)]);
    }
  };
  append_mel(cfg.start_mel_token, 0);

  std::vector<int64_t> codes;
  for (int64_t step = 0; step < cfg.max_mel_tokens; ++step) {
    const std::vector<float> hidden = gpt2::ForwardHostEmbeds(params, backbone, seq);
    const int64_t rows = static_cast<int64_t>(seq.size()) / dim;
    const size_t last = static_cast<size_t>((rows - 1) * dim);

    // Sequential(final_norm, mel_head): LayerNorm then Linear.
    double mean = 0.0;
    for (int64_t d = 0; d < dim; ++d) {
      mean += static_cast<double>(hidden[last + static_cast<size_t>(d)]);
    }
    mean /= static_cast<double>(dim);
    double var = 0.0;
    for (int64_t d = 0; d < dim; ++d) {
      const double delta = static_cast<double>(hidden[last + static_cast<size_t>(d)]) - mean;
      var += delta * delta;
    }
    var /= static_cast<double>(dim);
    const double inv = 1.0 / std::sqrt(var + cfg.layer_norm_eps);

    std::vector<double> normed(static_cast<size_t>(dim));
    for (int64_t d = 0; d < dim; ++d) {
      normed[static_cast<size_t>(d)] =
          ((static_cast<double>(hidden[last + static_cast<size_t>(d)]) - mean) * inv) *
              static_cast<double>(w.final_norm_w[static_cast<size_t>(d)]) +
          static_cast<double>(w.final_norm_b[static_cast<size_t>(d)]);
    }

    int64_t best = 0;
    double best_logit = 0.0;
    for (int64_t o = 0; o < cfg.mel_codes; ++o) {
      double acc = w.mel_head_b.empty()
                       ? 0.0
                       : static_cast<double>(w.mel_head_b[static_cast<size_t>(o)]);
      for (int64_t d = 0; d < dim; ++d) {
        acc += normed[static_cast<size_t>(d)] *
               static_cast<double>(w.mel_head_w[static_cast<size_t>(o * dim + d)]);
      }
      // Strictly greater, so ties keep the LOWEST index -- argmax's convention.
      if (o == 0 || acc > best_logit) {
        best_logit = acc;
        best = o;
      }
    }

    if (best == cfg.stop_mel_token) {
      break;  // the stop token is NOT part of the output
    }
    codes.push_back(best);
    // The next row's position is step + 1: the start token already took 0.
    append_mel(best, step + 1);
  }
  return codes;
}

}  // namespace talker
}  // namespace models
}  // namespace vllm
