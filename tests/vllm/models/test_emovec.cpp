// The supplied emotion vector. See emovec.h.
//
// Hand-computed throughout: the operation is a cosine argmax, a row selection
// and a weighted sum, so values chosen to be distinguishable prove it exactly.
#include <cstdint>
#include <vector>

#include "doctest/doctest.h"
#include "vllm/model_executor/models/emovec.h"

namespace ev = vllm::models::emovec;

namespace {

// Two emotions, two rows each, style_dim 2, out_dim 2.
//   bank 0 speakers: row0 = (1, 0), row1 = (0, 1)
//   bank 0 emotions: row0 = (10, 20), row1 = (30, 40)
//   bank 1 speakers: row0 = (0, 1), row1 = (1, 0)   <- REVERSED
//   bank 1 emotions: row0 = (1, 2), row1 = (3, 4)
std::vector<ev::EmotionBank> Banks() {
  ev::EmotionBank a;
  a.rows = 2;
  a.speakers = {1.0F, 0.0F, 0.0F, 1.0F};
  a.emotions = {10.0F, 20.0F, 30.0F, 40.0F};
  ev::EmotionBank b;
  b.rows = 2;
  b.speakers = {0.0F, 1.0F, 1.0F, 0.0F};
  b.emotions = {1.0F, 2.0F, 3.0F, 4.0F};
  return {a, b};
}

}  // namespace

TEST_CASE("each emotion searches ITS OWN speaker matrix") {
  // style = (1, 0) matches bank 0's row 0 and bank 1's row 1. One shared index
  // would pick the same row in both, which is the simplification this catches.
  std::vector<int64_t> chosen;
  const std::vector<float> out =
      ev::Select({1.0F, 0.0F}, 2, Banks(), {1.0F, 1.0F}, 2, &chosen);
  REQUIRE(chosen.size() == 2);
  CHECK(chosen[0] == 0);
  CHECK(chosen[1] == 1);
  // 1*(10,20) + 1*(3,4) = (13, 24)
  REQUIRE(out.size() == 2);
  CHECK(out[0] == 13.0F);
  CHECK(out[1] == 24.0F);
}

TEST_CASE("the weights scale each emotion's contribution") {
  const std::vector<float> out = ev::Select({1.0F, 0.0F}, 2, Banks(), {2.0F, 0.5F}, 2);
  // 2*(10,20) + 0.5*(3,4) = (21.5, 42)
  CHECK(out[0] == 21.5F);
  CHECK(out[1] == 42.0F);
}

TEST_CASE("a zero weight removes that emotion entirely") {
  const std::vector<float> out = ev::Select({1.0F, 0.0F}, 2, Banks(), {0.0F, 1.0F}, 2);
  CHECK(out[0] == 3.0F);
  CHECK(out[1] == 4.0F);
}

TEST_CASE("the match is COSINE, not Euclidean distance") {
  // A style of (5, 0) is far from (1, 0) in distance -- and further still from
  // (0, 1) -- but cosine sees only direction, so row 0 still wins. Scaling the
  // style must not change the selection at all.
  std::vector<int64_t> a;
  std::vector<int64_t> b;
  ev::Select({1.0F, 0.0F}, 2, Banks(), {1.0F, 1.0F}, 2, &a);
  ev::Select({5.0F, 0.0F}, 2, Banks(), {1.0F, 1.0F}, 2, &b);
  CHECK(a == b);

  // And a bank whose rows differ only in MAGNITUDE must be undecidable by
  // cosine, so the tie keeps the lower index.
  ev::EmotionBank same;
  same.rows = 2;
  same.speakers = {1.0F, 0.0F, 7.0F, 0.0F};  // same direction, different norms
  same.emotions = {1.0F, 1.0F, 9.0F, 9.0F};
  std::vector<int64_t> c;
  const std::vector<float> out = ev::Select({1.0F, 0.0F}, 2, {same}, {1.0F}, 2, &c);
  REQUIRE(c.size() == 1);
  CHECK(c[0] == 0);
  CHECK(out[0] == 1.0F);
}

TEST_CASE("a zero speaker row scores zero rather than NaN") {
  ev::EmotionBank z;
  z.rows = 2;
  z.speakers = {0.0F, 0.0F, 1.0F, 0.0F};
  z.emotions = {5.0F, 5.0F, 6.0F, 6.0F};
  std::vector<int64_t> chosen;
  const std::vector<float> out = ev::Select({1.0F, 0.0F}, 2, {z}, {1.0F}, 2, &chosen);
  CHECK(chosen[0] == 1);       // the real row wins over the degenerate one
  CHECK(out[0] == 6.0F);
}

TEST_CASE("mismatched shapes are refused") {
  auto banks = Banks();
  banks[0].emotions.pop_back();
  CHECK_THROWS(ev::Select({1.0F, 0.0F}, 2, banks, {1.0F, 1.0F}, 2));
  CHECK_THROWS(ev::Select({1.0F, 0.0F}, 2, Banks(), {1.0F}, 2));      // too few weights
  CHECK_THROWS(ev::Select({1.0F}, 2, Banks(), {1.0F, 1.0F}, 2));      // short style
}

// ---------------------------------------------------------------------------
// Loading the banks from the converted aux.safetensors (#634).
// ---------------------------------------------------------------------------
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>

namespace {

std::atomic<uint64_t> kUnique{0};

std::string U64Le(uint64_t v) {
  std::string out(8, '\0');
  for (int i = 0; i < 8; ++i) {
    out[static_cast<size_t>(i)] = static_cast<char>((v >> (8 * i)) & 0xFF);
  }
  return out;
}

// A minimal aux file: feat1 [rows, style], feat2 [rows, out].
std::string BuildAux(int64_t rows, int64_t style, int64_t out) {
  std::string header = "{";
  std::string data;
  auto add = [&](const std::string& name, int64_t r, int64_t c, float base) {
    const size_t begin = data.size();
    for (int64_t i = 0; i < r * c; ++i) {
      const float v = base + static_cast<float>(i);
      data.append(reinterpret_cast<const char*>(&v), 4);
    }
    const size_t end = data.size();
    if (header.size() > 1) header += ",";
    header += "\"" + name + "\":{\"dtype\":\"F32\",\"shape\":[" + std::to_string(r) + "," +
              std::to_string(c) + "],\"data_offsets\":[" + std::to_string(begin) + "," +
              std::to_string(end) + "]}";
  };
  add("feat1", rows, style, 0.0F);
  add("feat2", rows, out, 1000.0F);
  header += "}";
  return U64Le(header.size()) + header + data;
}

std::string WriteAux(const std::string& bytes) {
  const std::filesystem::path p = std::filesystem::temp_directory_path() /
      ("indextts2_aux_" + std::to_string(kUnique.fetch_add(1)) + ".safetensors");
  std::ofstream f(p, std::ios::binary);
  f.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
  return p.string();
}

}  // namespace

TEST_CASE("the banks are SPLIT by emo_num, not left as one") {
  // 6 rows split [1, 2, 3]. Without the split there would be one bank and one
  // index, which is exactly the defect the selection cases above catch.
  const std::string path = WriteAux(BuildAux(6, 2, 3));
  int64_t style = 0, out = 0;
  const auto banks = ev::LoadBanks(path, {1, 2, 3}, &style, &out);
  REQUIRE(banks.size() == 3);
  CHECK(style == 2);
  CHECK(out == 3);
  CHECK(banks[0].rows == 1);
  CHECK(banks[1].rows == 2);
  CHECK(banks[2].rows == 3);
  // Bank 1 must start at row 1 of feat1, i.e. value 2.0 (row 0 held 0, 1).
  CHECK(banks[1].speakers[0] == 2.0F);
  // Bank 2 starts at row 3 of feat2: 1000 + 3*3 = 1009.
  CHECK(banks[2].emotions[0] == 1009.0F);
  std::filesystem::remove(path);
}

TEST_CASE("an emo_num that does not sum to the row count is REFUSED") {
  const std::string path = WriteAux(BuildAux(6, 2, 3));
  int64_t style = 0, out = 0;
  // Sums to 5, not 6: the last row would be silently dropped and every bank
  // after the first would address the wrong rows.
  CHECK_THROWS(ev::LoadBanks(path, {1, 2, 2}, &style, &out));
  // Sums to 7: reads past the end.
  CHECK_THROWS(ev::LoadBanks(path, {1, 2, 4}, &style, &out));
  std::filesystem::remove(path);
}

TEST_CASE("feat1 and feat2 disagreeing on row count is REFUSED") {
  // The two matrices are indexed by the SAME row, so a mismatch means one of
  // them is addressed out of step with the other. Every shape stays valid and
  // the selection silently reads the wrong emotion rows.
  std::string header = "{";
  std::string data;
  auto add = [&](const std::string& name, int64_t r, int64_t c) {
    const size_t begin = data.size();
    for (int64_t i = 0; i < r * c; ++i) {
      const float v = static_cast<float>(i);
      data.append(reinterpret_cast<const char*>(&v), 4);
    }
    if (header.size() > 1) header += ",";
    header += "\"" + name + "\":{\"dtype\":\"F32\",\"shape\":[" + std::to_string(r) +
              "," + std::to_string(c) + "],\"data_offsets\":[" + std::to_string(begin) +
              "," + std::to_string(data.size()) + "]}";
  };
  add("feat1", 6, 2);
  add("feat2", 5, 3);  // one row SHORT
  header += "}";
  const std::string path = WriteAux(U64Le(header.size()) + header + data);
  int64_t style = 0, out = 0;
  CHECK_THROWS(ev::LoadBanks(path, {1, 2, 3}, &style, &out));
  std::filesystem::remove(path);
}

TEST_CASE("the SHIPPED aux file loads with the emo_num config.yaml declares") {
  const char* env = std::getenv("VLLM_CPP_INDEXTTS2_AUX");
  if (env == nullptr) {
    MESSAGE("SKIPPED: set VLLM_CPP_INDEXTTS2_AUX to the converted aux.safetensors");
    return;
  }
  int64_t style = 0, out = 0;
  const auto banks = ev::LoadBanks(std::string(env), {3, 17, 2, 8, 4, 5, 10, 24},
                                   &style, &out);
  CHECK(banks.size() == 8);
  CHECK(style == 192);   // kStyleDim
  CHECK(out == 1280);    // kTalkerDim
  int64_t rows = 0;
  for (const auto& b : banks) {
    rows += b.rows;
  }
  CHECK(rows == 73);
}
