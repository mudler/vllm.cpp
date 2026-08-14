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

// ---------------------------------------------------------------------------
// prepare_gpt_inputs: what the backbone sees before the first mel code (#634).
//
// Pinned by HAND-COMPUTED cases rather than a golden. The operation is embedding
// lookups, three adds and a pad, so values chosen to be distinguishable prove
// the layout exactly, and more legibly than a table of floats would.
// ---------------------------------------------------------------------------

namespace {

vllm::models::talker::PromptConfig PromptCfg() {
  vllm::models::talker::PromptConfig c;
  c.dim = 2;
  c.start_text_token = 0;
  c.stop_text_token = 1;
  c.start_mel_token = 777;
  c.text_slots = 4;
  return c;
}

// text_embedding[t] = (100t, 100t); positions = (10i, 10i); languages = (l, l).
vllm::models::talker::PromptWeights PromptW() {
  vllm::models::talker::PromptWeights w;
  for (int64_t t = 0; t < 8; ++t) {
    w.text_embedding.push_back(static_cast<float>(100 * t));
    w.text_embedding.push_back(static_cast<float>(100 * t));
  }
  for (int64_t i = 0; i < 8; ++i) {
    w.text_pos_embedding.push_back(static_cast<float>(10 * i));
    w.text_pos_embedding.push_back(static_cast<float>(10 * i));
  }
  for (int64_t l = 0; l < 4; ++l) {
    w.lang_embedding.push_back(static_cast<float>(l));
    w.lang_embedding.push_back(static_cast<float>(l));
  }
  return w;
}

}  // namespace

TEST_CASE("the prompt is [pad][conditioning][text], pad FIRST") {
  // 3 conditioning rows, 4 text slots -> target_len 9. The text {5, 6} becomes
  // [start=0, 5, 6, stop=1], so n = 4 and padding = 4 + 2 - 4 = 2.
  const std::vector<float> cond{-1.0F, -1.0F, -2.0F, -2.0F, -3.0F, -3.0F};
  const auto p = vllm::models::talker::PrepareInputs(PromptCfg(), PromptW(), cond, 3,
                                                     {5, 6}, /*lang=*/2);
  REQUIRE(p.target_len == 9);
  REQUIRE(p.embeds.size() == 18);

  // Rows 0-1 are the pad, and they are ZERO.
  for (int i = 0; i < 4; ++i) {
    CHECK(p.embeds[static_cast<size_t>(i)] == 0.0F);
  }
  // Rows 2-4 are the conditioning, UNCHANGED and AFTER the pad.
  CHECK(p.embeds[4] == -1.0F);
  CHECK(p.embeds[6] == -2.0F);
  CHECK(p.embeds[8] == -3.0F);
  // Row 5 is the start_text token at position 0 with language 2:
  //   100*0 + 10*0 + 2 = 2.
  CHECK(p.embeds[10] == 2.0F);
  // Row 6 is token 5 at position 1: 500 + 10 + 2 = 512.
  CHECK(p.embeds[12] == 512.0F);
  // Row 7 is token 6 at position 2: 600 + 20 + 2 = 622.
  CHECK(p.embeds[14] == 622.0F);
  // Row 8 is the stop_text token at position 3: 100 + 30 + 2 = 132.
  CHECK(p.embeds[16] == 132.0F);
}

TEST_CASE("the mask is one longer than the embeddings and masks only the pad") {
  const std::vector<float> cond(6, 0.0F);
  const auto p = vllm::models::talker::PrepareInputs(PromptCfg(), PromptW(), cond, 3,
                                                     {5, 6}, 0);
  REQUIRE(p.attention_mask.size() == 10);  // target_len 9 + the start_mel slot
  CHECK(p.attention_mask[0] == 0);
  CHECK(p.attention_mask[1] == 0);
  for (size_t i = 2; i < p.attention_mask.size(); ++i) {
    CHECK(p.attention_mask[i] == 1);  // the conditioning is NEVER masked
  }
}

TEST_CASE("the last input id is the start_mel_token") {
  const std::vector<float> cond(6, 0.0F);
  const auto p = vllm::models::talker::PrepareInputs(PromptCfg(), PromptW(), cond, 3,
                                                     {5, 6}, 0);
  REQUIRE(p.input_ids.size() == 10);
  CHECK(p.input_ids.back() == 777);
  for (size_t i = 0; i + 1 < p.input_ids.size(); ++i) {
    CHECK(p.input_ids[i] == 1);
  }
}

TEST_CASE("delimiters already in the input are STRIPPED, not doubled") {
  const std::vector<float> cond(6, 0.0F);
  const auto bare = vllm::models::talker::PrepareInputs(PromptCfg(), PromptW(), cond, 3,
                                                        {5, 6}, 0);
  // Same text, already delimited. It must produce an IDENTICAL prompt.
  const auto delim = vllm::models::talker::PrepareInputs(PromptCfg(), PromptW(), cond, 3,
                                                         {0, 5, 6, 1}, 0);
  CHECK(delim.embeds == bare.embeds);
  CHECK(delim.attention_mask == bare.attention_mask);
}

TEST_CASE("the language vector reaches EVERY text position") {
  const std::vector<float> cond(6, 0.0F);
  const auto a = vllm::models::talker::PrepareInputs(PromptCfg(), PromptW(), cond, 3,
                                                     {5, 6}, 1);
  const auto b = vllm::models::talker::PrepareInputs(PromptCfg(), PromptW(), cond, 3,
                                                     {5, 6}, 3);
  // Language 3 minus language 1 is +2 on every text row, and 0 elsewhere.
  for (int64_t row = 0; row < a.target_len; ++row) {
    const double diff = b.embeds[static_cast<size_t>(row * 2)] -
                        a.embeds[static_cast<size_t>(row * 2)];
    if (row >= 5) {
      CHECK(diff == doctest::Approx(2.0));
    } else {
      CHECK(diff == doctest::Approx(0.0));
    }
  }
}

TEST_CASE("text longer than the declared slots is REFUSED") {
  const std::vector<float> cond(6, 0.0F);
  CHECK_THROWS(vllm::models::talker::PrepareInputs(PromptCfg(), PromptW(), cond, 3,
                                                   {2, 3, 4, 5, 6, 7}, 0));
}

// ---------------------------------------------------------------------------
// The autoregressive loop (#634).
//
// Driven by weights built so the argmax is PREDICTABLE: the mel head is an
// identity-like projection, so whichever hidden dimension is largest names the
// code. That makes the loop's control flow -- stop token, length cap, position
// indexing -- checkable by hand rather than against a table of floats.
// ---------------------------------------------------------------------------

namespace {

// A backbone that passes its input through unchanged: zero attention output,
// zero MLP output, and identity layer norms are impossible, so instead every
// weight is zero EXCEPT the layer norms, which makes each block's residual
// dominate. What matters for these cases is only that the last row's hidden
// state is a deterministic function of the last input row.
vllm::gpt2::Params GenParams(int64_t dim) {
  vllm::gpt2::Params p;
  p.hidden_size = dim;
  p.num_attention_heads = 1;
  p.num_hidden_layers = 1;
  p.inner_size = dim;
  p.vocab_size = 8;
  p.max_position_embeddings = 8;
  p.layer_norm_eps = 1e-5;
  return p;
}

vllm::gpt2::Weights GenBackbone(int64_t dim) {
  vllm::gpt2::Weights w;
  w.wte.assign(static_cast<size_t>(8 * dim), 0.0F);
  w.wpe.assign(static_cast<size_t>(8 * dim), 0.0F);
  w.ln_f_weight.assign(static_cast<size_t>(dim), 1.0F);
  w.ln_f_bias.assign(static_cast<size_t>(dim), 0.0F);
  vllm::gpt2::LayerWeights l;
  l.ln_1_weight.assign(static_cast<size_t>(dim), 1.0F);
  l.ln_1_bias.assign(static_cast<size_t>(dim), 0.0F);
  l.ln_2_weight.assign(static_cast<size_t>(dim), 1.0F);
  l.ln_2_bias.assign(static_cast<size_t>(dim), 0.0F);
  l.c_attn_weight.assign(static_cast<size_t>(3 * dim * dim), 0.0F);
  l.c_attn_bias.assign(static_cast<size_t>(3 * dim), 0.0F);
  l.c_proj_weight.assign(static_cast<size_t>(dim * dim), 0.0F);
  l.c_proj_bias.assign(static_cast<size_t>(dim), 0.0F);
  l.c_fc_weight.assign(static_cast<size_t>(dim * dim), 0.0F);
  l.c_fc_bias.assign(static_cast<size_t>(dim), 0.0F);
  l.mlp_c_proj_weight.assign(static_cast<size_t>(dim * dim), 0.0F);
  l.mlp_c_proj_bias.assign(static_cast<size_t>(dim), 0.0F);
  w.layers.push_back(std::move(l));
  return w;
}

vllm::models::talker::GenerateConfig GenCfg(int64_t codes, int64_t stop, int64_t maxlen) {
  vllm::models::talker::GenerateConfig c;
  c.dim = 2;
  c.mel_codes = codes;
  c.start_mel_token = codes - 1;
  c.stop_mel_token = stop;
  c.max_mel_tokens = maxlen;
  return c;
}

// The head selects `bias_favoured` unconditionally: every head row is zero and
// only that row carries a positive bias.
vllm::models::talker::GenerateWeights GenW(int64_t codes, int64_t dim, int64_t favoured) {
  vllm::models::talker::GenerateWeights w;
  w.mel_embedding.assign(static_cast<size_t>(codes * dim), 0.0F);
  w.mel_pos_embedding.assign(static_cast<size_t>(64 * dim), 0.0F);
  w.final_norm_w.assign(static_cast<size_t>(dim), 1.0F);
  w.final_norm_b.assign(static_cast<size_t>(dim), 0.0F);
  w.mel_head_w.assign(static_cast<size_t>(codes * dim), 0.0F);
  w.mel_head_b.assign(static_cast<size_t>(codes), 0.0F);
  w.mel_head_b[static_cast<size_t>(favoured)] = 1.0F;
  return w;
}

}  // namespace

TEST_CASE("generation stops at the stop token, which is NOT emitted") {
  const int64_t dim = 2, codes = 5;
  // The head always favours code 3, and 3 IS the stop token, so nothing is
  // emitted at all.
  const auto out = vllm::models::talker::GenerateMelCodes(
      GenCfg(codes, /*stop=*/3, /*maxlen=*/8), GenW(codes, dim, 3), GenParams(dim),
      GenBackbone(dim), std::vector<float>(static_cast<size_t>(3 * dim), 0.1F), 3);
  CHECK(out.empty());
}

TEST_CASE("generation is capped by max_mel_tokens") {
  const int64_t dim = 2, codes = 5;
  // The head always favours code 2, which is NOT the stop token, so the loop
  // runs to the cap.
  const auto out = vllm::models::talker::GenerateMelCodes(
      GenCfg(codes, /*stop=*/3, /*maxlen=*/6), GenW(codes, dim, 2), GenParams(dim),
      GenBackbone(dim), std::vector<float>(static_cast<size_t>(3 * dim), 0.1F), 3);
  REQUIRE(out.size() == 6);
  for (const int64_t c : out) {
    CHECK(c == 2);
  }
}

TEST_CASE("an out-of-range mel code is refused rather than indexed") {
  const int64_t dim = 2, codes = 5;
  auto w = GenW(codes, dim, 2);
  w.mel_embedding.resize(static_cast<size_t>(2 * dim));  // too short for code 2
  CHECK_THROWS(vllm::models::talker::GenerateMelCodes(
      GenCfg(codes, 3, 4), w, GenParams(dim), GenBackbone(dim),
      std::vector<float>(static_cast<size_t>(3 * dim), 0.1F), 3));
}

TEST_CASE("a prompt whose length disagrees with its rows is refused") {
  const int64_t dim = 2, codes = 5;
  CHECK_THROWS(vllm::models::talker::GenerateMelCodes(
      GenCfg(codes, 3, 4), GenW(codes, dim, 2), GenParams(dim), GenBackbone(dim),
      std::vector<float>(static_cast<size_t>(3 * dim), 0.1F), /*prompt_len=*/4));
}

TEST_CASE("the mel POSITION table is indexed by step, not absolute position") {
  // A position table only long enough for the generated run proves the index is
  // the step: an absolute index would run off the end of it and throw.
  const int64_t dim = 2, codes = 5;
  auto w = GenW(codes, dim, 2);
  w.mel_pos_embedding.assign(static_cast<size_t>(5 * dim), 0.0F);  // 4 steps + start
  const auto out = vllm::models::talker::GenerateMelCodes(
      GenCfg(codes, 3, 4), w, GenParams(dim), GenBackbone(dim),
      std::vector<float>(static_cast<size_t>(16 * dim), 0.1F), 16);
  CHECK(out.size() == 4);
}

TEST_CASE("the START token is SEEDED, and the loop sees it") {
  // The all-zero fixtures above cannot tell a seeded run from an unseeded one:
  // with a zero backbone and a bias-only head, both predict identically. So
  // this one makes the prediction depend on the LAST ROW. The start token
  // embeds to a large negative value while the prompt rows are positive, and
  // the head reads dimension 0, so whether the start row was appended decides
  // which code wins.
  const int64_t dim = 2, codes = 5;
  auto w = GenW(codes, dim, 0);
  w.mel_head_b.assign(static_cast<size_t>(codes), 0.0F);
  // code 2 fires on a POSITIVE dim-0, code 1 on a negative one.
  w.mel_head_w.assign(static_cast<size_t>(codes * dim), 0.0F);
  w.mel_head_w[static_cast<size_t>(2 * dim)] = 1.0F;
  w.mel_head_w[static_cast<size_t>(1 * dim)] = -1.0F;
  // The start token (code 4) embeds to a strong negative on dim 0.
  w.mel_embedding.assign(static_cast<size_t>(codes * dim), 0.0F);
  w.mel_embedding[static_cast<size_t>(4 * dim)] = -50.0F;

  const std::vector<float> prompt{5.0F, -5.0F, 5.0F, -5.0F, 5.0F, -5.0F};
  const auto out = vllm::models::talker::GenerateMelCodes(
      GenCfg(codes, /*stop=*/3, /*maxlen=*/1), w, GenParams(dim), GenBackbone(dim),
      prompt, 3);
  REQUIRE(out.size() == 1);
  // Layer-norming a row whose dim 0 is the NEGATIVE start value leaves dim 0
  // negative, so code 1 wins. Without the seeding the last row would be a
  // prompt row with a POSITIVE dim 0, and code 2 would win instead.
  CHECK(out[0] == 1);
}

TEST_CASE("argmax ties keep the LOWEST code") {
  // Two codes with identical logits. Upstream's argmax returns the first, and a
  // `>=` comparison silently returns the last -- both produce a legal code, and
  // on a real checkpoint exact ties are rare enough that a golden would never
  // catch it.
  const int64_t dim = 2, codes = 5;
  auto w = GenW(codes, dim, 1);
  w.mel_head_b[4] = 1.0F;  // code 1 and code 4 now tie
  const auto out = vllm::models::talker::GenerateMelCodes(
      GenCfg(codes, /*stop=*/3, /*maxlen=*/2), w, GenParams(dim), GenBackbone(dim),
      std::vector<float>(static_cast<size_t>(3 * dim), 0.1F), 3);
  REQUIRE(out.size() == 2);
  CHECK(out[0] == 1);
  CHECK(out[1] == 1);
}
