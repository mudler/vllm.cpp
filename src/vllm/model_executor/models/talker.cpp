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

}  // namespace talker
}  // namespace models
}  // namespace vllm
