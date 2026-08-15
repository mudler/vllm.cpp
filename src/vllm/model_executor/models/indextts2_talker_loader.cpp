// Talker checkpoint binding. See indextts2_talker_loader.h for the anchors.
#include "vllm/model_executor/models/indextts2_talker_loader.h"

#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace vllm {
namespace models {
namespace indextts2 {
namespace {

[[noreturn]] void Fail(const std::string& what) {
  throw std::runtime_error("IndexTTS-2.5 talker: " + what);
}

int64_t Elems(const std::vector<int64_t>& shape) {
  int64_t n = 1;
  for (const int64_t d : shape) {
    n *= d;
  }
  return n;
}

const StTensor* Find(const SafetensorsFile& file, const std::string& name) {
  try {
    return &file.Get(name);
  } catch (const std::exception&) {
    Fail("missing tensor '" + name + "'");
  }
}

std::vector<float> Read(const SafetensorsFile& file, const std::string& name) {
  const StTensor* t = Find(file, name);
  if (t->dtype != "F32") {
    Fail("tensor '" + name + "' is " + t->dtype + ", expected F32");
  }
  const int64_t n = Elems(t->shape);
  if (t->nbytes != static_cast<size_t>(n) * sizeof(float)) {
    Fail("tensor '" + name + "' byte length disagrees with its shape");
  }
  std::vector<float> out(static_cast<size_t>(n));
  std::memcpy(out.data(), t->data, t->nbytes);
  return out;
}

std::vector<int64_t> Shape(const SafetensorsFile& file, const std::string& name) {
  return Find(file, name)->shape;
}

int64_t CountLayers(const SafetensorsFile& file) {
  int64_t n = 0;
  for (const std::string& name : file.Names()) {
    if (name.rfind("gpt.h.", 0) == 0 && name.find(".attn.c_attn.weight") != std::string::npos) {
      ++n;
    }
  }
  return n;
}

}  // namespace

TalkerWeights LoadTalker(const SafetensorsFile& file, int64_t num_attention_heads) {
  TalkerWeights out;

  const std::vector<int64_t> text_emb = Shape(file, "text_embedding.weight");
  if (text_emb.size() != 2) {
    Fail("text_embedding.weight must be 2-D");
  }
  out.text_vocab = text_emb[0];
  const int64_t hidden = text_emb[1];

  const std::vector<int64_t> mel_emb = Shape(file, "mel_embedding.weight");
  if (mel_emb.size() != 2 || mel_emb[1] != hidden) {
    Fail("mel_embedding.weight must be [mel_codes, hidden]");
  }
  out.mel_codes = mel_emb[0];

  const std::vector<int64_t> tpos = Shape(file, "text_pos_embedding.emb.weight");
  const std::vector<int64_t> mpos = Shape(file, "mel_pos_embedding.emb.weight");
  if (tpos.size() != 2 || mpos.size() != 2) {
    Fail("the position tables must be 2-D");
  }
  out.text_positions = tpos[0];
  out.mel_positions = mpos[0];

  const int64_t layers = CountLayers(file);
  if (layers <= 0) {
    Fail("no gpt.h.N blocks found; this is not a talker checkpoint");
  }

  // c_attn is [H, 3H] in the file (GPT-2 Conv1D orientation), so the head count
  // is not recoverable from it. It comes from the config, and the ONE thing the
  // checkpoint can confirm is that 3H divides as expected.
  const std::vector<int64_t> cattn = Shape(file, "gpt.h.0.attn.c_attn.weight");
  if (cattn.size() != 2 || cattn[0] != hidden || cattn[1] != 3 * hidden) {
    Fail("gpt.h.0.attn.c_attn.weight must be [hidden, 3 * hidden]");
  }
  const std::vector<int64_t> cfc = Shape(file, "gpt.h.0.mlp.c_fc.weight");
  if (cfc.size() != 2 || cfc[0] != hidden) {
    Fail("gpt.h.0.mlp.c_fc.weight must be [hidden, inner]");
  }

  out.params.hidden_size = hidden;
  out.params.num_hidden_layers = layers;
  out.params.inner_size = cfc[1];
  out.params.vocab_size = out.text_vocab;
  out.params.max_position_embeddings = out.text_positions;
  // NOT validated here. `gpt2::Load` already requires num_attention_heads to be
  // positive and to divide hidden_size, and duplicating that would be a second
  // copy of a rule with one owner. Mutation testing is what surfaced this: the
  // duplicate could be deleted with every gate still green, which is the
  // signature of a redundant check rather than a weak test.
  out.params.num_attention_heads = num_attention_heads;

  // The blocks go through the SHARED gpt2 loader, which owns the Conv1D
  // transpose. Only the naming is talker-specific.
  gpt2::CheckpointTensors ct;
  auto put = [&](const std::string& dst, const std::string& src) {
    ct.Set(dst, Shape(file, src), Read(file, src));
  };
  for (int64_t i = 0; i < layers; ++i) {
    const std::string s = "gpt.h." + std::to_string(i) + ".";
    const std::string d = "h." + std::to_string(i) + ".";
    for (const char* leaf : {"ln_1.weight", "ln_1.bias", "ln_2.weight", "ln_2.bias",
                             "attn.c_attn.weight", "attn.c_attn.bias",
                             "attn.c_proj.weight", "attn.c_proj.bias",
                             "mlp.c_fc.weight", "mlp.c_fc.bias",
                             "mlp.c_proj.weight", "mlp.c_proj.bias"}) {
      put(d + leaf, s + leaf);
    }
  }
  put("ln_f.weight", "gpt.ln_f.weight");
  put("ln_f.bias", "gpt.ln_f.bias");
  // The talker has NO wte/wpe. The shared loader expects them, so it is handed
  // the talker's TEXT tables: they are the same shape and the same role, and
  // the mel stream is carried separately below.
  ct.Set("wte.weight", text_emb, Read(file, "text_embedding.weight"));
  ct.Set("wpe.weight", tpos, Read(file, "text_pos_embedding.emb.weight"));

  out.backbone = gpt2::Load(out.params, ct);

  out.text_embedding = Read(file, "text_embedding.weight");
  out.mel_embedding = Read(file, "mel_embedding.weight");
  out.text_pos_embedding = Read(file, "text_pos_embedding.emb.weight");
  out.mel_pos_embedding = Read(file, "mel_pos_embedding.emb.weight");
  out.final_norm_w = Read(file, "final_norm.weight");
  out.final_norm_b = Read(file, "final_norm.bias");
  out.text_head_w = Read(file, "text_head.weight");
  out.text_head_b = Read(file, "text_head.bias");
  out.mel_head_w = Read(file, "mel_head.weight");
  out.mel_head_b = Read(file, "mel_head.bias");
  out.spk_emb_proj_w = Read(file, "spk_emb_proj.weight");
  out.spk_emb_proj_b = Read(file, "spk_emb_proj.bias");
  out.lang_embedding = Read(file, "lang_embedding.weight");

  const std::vector<int64_t> spk = Shape(file, "spk_emb_proj.weight");
  if (spk.size() != 2 || spk[0] != hidden) {
    Fail("spk_emb_proj.weight must be [hidden, style_dim]");
  }
  out.style_dim = spk[1];
  out.languages = Shape(file, "lang_embedding.weight")[0];

  // The mel head must cover every mel code, or a legal id has no logit.
  if (static_cast<int64_t>(out.mel_head_w.size()) != out.mel_codes * hidden) {
    Fail("mel_head.weight does not cover every mel code");
  }
  return out;
}

TalkerWeights LoadTalker(const std::string& path, int64_t num_attention_heads) {
  const SafetensorsFile file = SafetensorsFile::Open(path);
  return LoadTalker(file, num_attention_heads);
}

}  // namespace indextts2
}  // namespace models
}  // namespace vllm
