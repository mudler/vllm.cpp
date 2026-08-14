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

}  // namespace talker
}  // namespace models
}  // namespace vllm
