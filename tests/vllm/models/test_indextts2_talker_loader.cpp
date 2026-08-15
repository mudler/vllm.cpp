// Binding the converted talker checkpoint. See indextts2_talker_loader.h.
//
// The real-checkpoint cases run when VLLM_CPP_INDEXTTS2_GPT points at the
// converted gpt.safetensors, and skip LOUDLY otherwise.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "doctest/doctest.h"
#include "vllm/model_executor/models/indextts2_config.h"
#include "vllm/model_executor/models/indextts2_talker_loader.h"

namespace {

std::string U64Le(uint64_t v) {
  std::string out(8, '\0');
  for (int i = 0; i < 8; ++i) {
    out[static_cast<size_t>(i)] = static_cast<char>((v >> (8 * i)) & 0xFF);
  }
  return out;
}

struct Builder {
  std::string header = "{";
  std::string data;
  bool first = true;
  void Add(const std::string& name, const std::vector<int64_t>& shape,
           const std::string& dtype = "F32") {
    int64_t n = 1;
    for (const int64_t d : shape) {
      n *= d;
    }
    const size_t begin = data.size();
    for (int64_t i = 0; i < n; ++i) {
      const float v = 0.03125F * static_cast<float>((i % 11) - 5);
      data.append(reinterpret_cast<const char*>(&v), 4);
    }
    const size_t end = begin + static_cast<size_t>(n) * 4;
    if (!first) header += ",";
    first = false;
    header += "\"" + name + "\":{\"dtype\":\"" + dtype + "\",\"shape\":[";
    for (size_t i = 0; i < shape.size(); ++i) header += (i ? "," : "") + std::to_string(shape[i]);
    header += "],\"data_offsets\":[" + std::to_string(begin) + "," + std::to_string(end) + "]}";
  }
  std::string Finish() { return U64Le(header.size() + 1) + header + "}" + data; }
};

std::string BuildTalker(int64_t H = 8, int64_t L = 2, int64_t I = 16, int64_t TV = 20,
                        int64_t MC = 12, int64_t TP = 9, int64_t MP = 7, int64_t STY = 4,
                        int64_t LANG = 3) {
  Builder b;
  for (int64_t i = 0; i < L; ++i) {
    const std::string p = "gpt.h." + std::to_string(i) + ".";
    b.Add(p + "ln_1.weight", {H}); b.Add(p + "ln_1.bias", {H});
    b.Add(p + "ln_2.weight", {H}); b.Add(p + "ln_2.bias", {H});
    b.Add(p + "attn.c_attn.weight", {H, 3 * H}); b.Add(p + "attn.c_attn.bias", {3 * H});
    b.Add(p + "attn.c_proj.weight", {H, H}); b.Add(p + "attn.c_proj.bias", {H});
    b.Add(p + "mlp.c_fc.weight", {H, I}); b.Add(p + "mlp.c_fc.bias", {I});
    b.Add(p + "mlp.c_proj.weight", {I, H}); b.Add(p + "mlp.c_proj.bias", {H});
  }
  b.Add("gpt.ln_f.weight", {H}); b.Add("gpt.ln_f.bias", {H});
  b.Add("text_embedding.weight", {TV, H});
  b.Add("mel_embedding.weight", {MC, H});
  b.Add("text_pos_embedding.emb.weight", {TP, H});
  b.Add("mel_pos_embedding.emb.weight", {MP, H});
  b.Add("final_norm.weight", {H}); b.Add("final_norm.bias", {H});
  b.Add("text_head.weight", {TV, H}); b.Add("text_head.bias", {TV});
  b.Add("mel_head.weight", {MC, H}); b.Add("mel_head.bias", {MC});
  b.Add("spk_emb_proj.weight", {H, STY}); b.Add("spk_emb_proj.bias", {H});
  b.Add("lang_embedding.weight", {LANG, H});
  return b.Finish();
}

std::string WriteTemp(const std::string& bytes, const std::string& tag) {
  const std::filesystem::path p = std::filesystem::temp_directory_path() /
      ("indextts2_talker_" + tag + "_" + std::to_string(::getpid()) + ".safetensors");
  std::ofstream out(p, std::ios::binary);
  out.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
  return p.string();
}

}  // namespace

TEST_CASE("the talker's TWO vocabularies and TWO position tables are separate") {
  const std::string path = WriteTemp(BuildTalker(), "dims");
  const auto t = vllm::models::indextts2::LoadTalker(path, 2);
  CHECK(t.params.hidden_size == 8);
  CHECK(t.params.num_hidden_layers == 2);
  CHECK(t.params.inner_size == 16);
  CHECK(t.text_vocab == 20);
  CHECK(t.mel_codes == 12);
  CHECK(t.text_positions == 9);
  CHECK(t.mel_positions == 7);
  CHECK(t.style_dim == 4);
  CHECK(t.languages == 3);
  // Distinct tables, not one aliased twice.
  CHECK(t.text_embedding.size() == 20UL * 8UL);
  CHECK(t.mel_embedding.size() == 12UL * 8UL);
  CHECK(t.text_pos_embedding.size() == 9UL * 8UL);
  CHECK(t.mel_pos_embedding.size() == 7UL * 8UL);
  std::filesystem::remove(path);
}

TEST_CASE("a mel head that does not cover every mel code is refused") {
  Builder b;
  const int64_t H = 8;
  for (int64_t i = 0; i < 1; ++i) {
    const std::string p = "gpt.h.0.";
    b.Add(p + "ln_1.weight", {H}); b.Add(p + "ln_1.bias", {H});
    b.Add(p + "ln_2.weight", {H}); b.Add(p + "ln_2.bias", {H});
    b.Add(p + "attn.c_attn.weight", {H, 3 * H}); b.Add(p + "attn.c_attn.bias", {3 * H});
    b.Add(p + "attn.c_proj.weight", {H, H}); b.Add(p + "attn.c_proj.bias", {H});
    b.Add(p + "mlp.c_fc.weight", {H, 16}); b.Add(p + "mlp.c_fc.bias", {16});
    b.Add(p + "mlp.c_proj.weight", {16, H}); b.Add(p + "mlp.c_proj.bias", {H});
  }
  b.Add("gpt.ln_f.weight", {H}); b.Add("gpt.ln_f.bias", {H});
  b.Add("text_embedding.weight", {20, H});
  b.Add("mel_embedding.weight", {12, H});
  b.Add("text_pos_embedding.emb.weight", {9, H});
  b.Add("mel_pos_embedding.emb.weight", {7, H});
  b.Add("final_norm.weight", {H}); b.Add("final_norm.bias", {H});
  b.Add("text_head.weight", {20, H}); b.Add("text_head.bias", {20});
  b.Add("mel_head.weight", {11, H});  // one row SHORT of mel_codes
  b.Add("mel_head.bias", {11});
  b.Add("spk_emb_proj.weight", {H, 4}); b.Add("spk_emb_proj.bias", {H});
  b.Add("lang_embedding.weight", {3, H});
  const std::string path = WriteTemp(b.Finish(), "shorthead");
  CHECK_THROWS_AS(vllm::models::indextts2::LoadTalker(path, 2), std::runtime_error);
  std::filesystem::remove(path);
}

TEST_CASE("a head count that does not divide the hidden size is REFUSED") {
  // The checkpoint cannot express the head count, so it arrives from the config
  // and is the one dimension nothing in the file can cross-check. The least it
  // must do is divide the hidden size; otherwise every head boundary is wrong
  // and the tower still runs.
  //
  // The rule lives in `gpt2::Load`, not here -- see the loader header. What
  // this pins is that the guarantee HOLDS through this entry point and that the
  // refusal names the number the caller got wrong, wherever it is raised.
  const std::string path = WriteTemp(BuildTalker(), "heads");  // hidden 8
  auto refused_by_us = [&](int64_t heads) {
    try {
      vllm::models::indextts2::LoadTalker(path, heads);
    } catch (const std::runtime_error& e) {
      return std::string(e.what()).find("num_attention_heads") != std::string::npos;
    }
    return false;
  };
  CHECK(refused_by_us(3));
  CHECK(refused_by_us(0));
  CHECK(refused_by_us(-1));
  // 4 divides 8, so this one must load.
  CHECK_NOTHROW(vllm::models::indextts2::LoadTalker(path, 4));
  std::filesystem::remove(path);
}

TEST_CASE("a file that is not a talker checkpoint is refused") {
  Builder b;
  b.Add("something.else", {4, 4});
  const std::string path = WriteTemp(b.Finish(), "wrong");
  CHECK_THROWS_AS(vllm::models::indextts2::LoadTalker(path, 2), std::runtime_error);
  std::filesystem::remove(path);
}

TEST_CASE("the SHIPPED talker loads, and the CONFIG disagrees with it") {
  const char* env = std::getenv("VLLM_CPP_INDEXTTS2_GPT");
  if (env == nullptr) {
    MESSAGE("SKIPPED: set VLLM_CPP_INDEXTTS2_GPT to the converted "
            "gpt.safetensors to check the real checkpoint");
    return;
  }
  const auto t = vllm::models::indextts2::LoadTalker(std::string(env), vllm::models::indextts2::kTalkerHeads);
  CHECK(t.params.hidden_size == vllm::models::indextts2::kTalkerDim);
  CHECK(t.params.num_hidden_layers == vllm::models::indextts2::kTalkerLayers);
  CHECK(t.params.inner_size == 4 * vllm::models::indextts2::kTalkerDim);
  CHECK(t.mel_codes == vllm::models::indextts2::kNumberMelCodes);
  CHECK(t.style_dim == vllm::models::indextts2::kStyleDim);

  // The disagreements, asserted so they cannot pass unnoticed. Both tables are
  // LARGER than the config claims -- the safe direction, but a port sized from
  // the config would be a row short on the last legal id.
  CHECK(t.text_vocab == vllm::models::indextts2::kNumberTextTokens + 1);
  CHECK(t.mel_positions == vllm::models::indextts2::kMaxMelTokens + 3);
  CHECK(t.languages == 107);
  CHECK(t.params.num_attention_heads == vllm::models::indextts2::kTalkerHeads);
}

TEST_CASE("the SHIPPED talker weights actually RUN through the ported backbone") {
  const char* env = std::getenv("VLLM_CPP_INDEXTTS2_GPT");
  if (env == nullptr) {
    MESSAGE("SKIPPED: no VLLM_CPP_INDEXTTS2_GPT, so the real talker was never run");
    return;
  }
  const auto t = vllm::models::indextts2::LoadTalker(
      std::string(env), vllm::models::indextts2::kTalkerHeads);

  // A short TEXT prompt through the 24-layer backbone. Ids are kept well inside
  // the table and the positions inside the text table, so this exercises the
  // tower rather than the bounds.
  const std::vector<int64_t> ids{1, 42, 7, 1900, 55, 3};
  std::vector<int64_t> pos(ids.size());
  for (size_t i = 0; i < pos.size(); ++i) {
    pos[i] = static_cast<int64_t>(i);
  }
  REQUIRE(static_cast<int64_t>(ids.size()) <= t.text_positions);

  const std::vector<float> hidden =
      vllm::gpt2::ForwardHost(t.params, t.backbone, ids, pos);
  REQUIRE(hidden.size() == ids.size() * static_cast<size_t>(t.params.hidden_size));

  float lo = hidden[0];
  float hi = hidden[0];
  for (const float v : hidden) {
    REQUIRE(std::isfinite(v));
    lo = std::min(lo, v);
    hi = std::max(hi, v);
  }
  CHECK(hi > lo);

  // Different ids must give different hidden states: a backbone that ignored
  // its input would still return finite, correctly shaped values.
  std::vector<int64_t> other = ids;
  other[2] = 900;
  const std::vector<float> hidden2 =
      vllm::gpt2::ForwardHost(t.params, t.backbone, other, pos);
  bool differs = false;
  for (size_t i = 0; i < hidden.size(); ++i) {
    if (hidden[i] != hidden2[i]) {
      differs = true;
    }
  }
  CHECK(differs);
  MESSAGE("real talker backbone ran: " << ids.size() << " tokens x "
          << t.params.hidden_size << ", range [" << lo << ", " << hi << "]");
}
