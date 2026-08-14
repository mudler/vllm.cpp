// IndexTTS-2.5 talker embedding scaffolding (#634).
//
// No golden file here, deliberately: this layer is pure INDEXING, and a captured
// tensor would only restate the table it was captured from. What can go wrong is
// WHICH row is read, so the cases assert that directly.
#include <cstdint>
#include <stdexcept>
#include <vector>

#include "doctest/doctest.h"
#include "vllm/model_executor/models/talker.h"

namespace {
// A table whose row r is filled with r*10 + d, so a row's identity is readable
// from any single element.
std::vector<float> Table(int64_t rows, int64_t dim) {
  std::vector<float> t(static_cast<size_t>(rows * dim));
  for (int64_t r = 0; r < rows; ++r) {
    for (int64_t d = 0; d < dim; ++d) {
      t[static_cast<size_t>(r * dim + d)] = static_cast<float>(r * 10 + d);
    }
  }
  return t;
}
}  // namespace

TEST_CASE("talker position rows are 0..seq_len-1 in order") {
  const std::vector<float> table = Table(8, 3);
  const std::vector<float> got = vllm::models::talker::PositionRows(table, 3, 4);
  REQUIRE(got.size() == 12U);
  for (int64_t t = 0; t < 4; ++t) {
    for (int64_t d = 0; d < 3; ++d) {
      CHECK(got[static_cast<size_t>(t * 3 + d)] == static_cast<float>(t * 10 + d));
    }
  }
}

TEST_CASE("talker get_fixed_embedding reads the row at the CURRENT step") {
  // The incremental-decode path. At step n the position is n; returning row 0
  // every step makes every generated frame believe it is the first, which still
  // decodes to audio and destroys the prosody.
  const std::vector<float> table = Table(8, 3);
  for (int64_t step = 0; step < 8; ++step) {
    const std::vector<float> row = vllm::models::talker::PositionRowAt(table, 3, step);
    REQUIRE(row.size() == 3U);
    CHECK(row[0] == static_cast<float>(step * 10));
    CHECK(row[2] == static_cast<float>(step * 10 + 2));
  }
}

TEST_CASE("talker refuses a position past the end of the table") {
  // `text_pos_embedding.emb.num_embeddings` is the talker's real capacity limit
  // (infer_v2_5.py:427 reads it to bound a request), so running off the end must
  // throw rather than read adjacent memory.
  const std::vector<float> table = Table(4, 3);
  CHECK_THROWS_AS(vllm::models::talker::PositionRowAt(table, 3, 4), std::runtime_error);
  CHECK_THROWS_AS(vllm::models::talker::PositionRows(table, 3, 5), std::runtime_error);
}

TEST_CASE("talker embedding ADDS the position row to the token row") {
  // Token embedding + learned position, which is on TOP of the GPT-2 backbone's
  // own wpe -- two tables, not one.
  const std::vector<float> tokens_table = Table(5, 3);
  const std::vector<float> pos_table = Table(5, 3);
  const std::vector<int64_t> ids{4, 0, 2};
  const std::vector<float> got =
      vllm::models::talker::EmbedWithPositions(ids, tokens_table, pos_table, 3, 5);
  REQUIRE(got.size() == 9U);
  // position 0 holds token 4: (40+d) + (0+d)
  CHECK(got[0] == 40.0F);
  CHECK(got[1] == 42.0F);   // 41 + 1
  // position 1 holds token 0: (0+d) + (10+d)
  CHECK(got[3] == 10.0F);
  CHECK(got[5] == 14.0F);   // 2 + 12
  // position 2 holds token 2: (20+d) + (20+d)
  CHECK(got[6] == 40.0F);
}

TEST_CASE("talker rejects an out-of-range token id") {
  const std::vector<float> table = Table(5, 3);
  const std::vector<int64_t> bad{5};
  CHECK_THROWS_AS(vllm::models::talker::EmbedWithPositions(bad, table, table, 3, 5),
                  std::runtime_error);
}
