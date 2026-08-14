// Bind the converted talker checkpoint to the ported GPT-2 backbone (#634).
//
// Reads `gpt.safetensors`, produced offline by
// `scripts/convert-indextts2-checkpoint.py` from upstream's `gpt.pth`.
//
// The talker is a GPT-2 BACKBONE with IndexTTS-specific heads bolted on, and
// the delta matters: there is NO `wte` and NO `wpe`. Text and mel each get their
// own embedding table and their own learned position table, because the talker
// interleaves two token streams. `gpt2::Load` still does the Conv1D transpose
// for the blocks; what this adds is the surrounding vocabulary.
//
// THE CONFIG AND THE CHECKPOINT DISAGREE, and the checkpoint wins. `config.yaml`
// says `number_text_tokens: 60509` and `max_mel_tokens: 1815`, but the shipped
// tables are [60510, 1280] and [1818, 1280]. Both are LARGER, which is the safe
// direction -- a table indexed by a token id that the config says cannot occur
// still has a row -- but a port that allocated from the config would be one row
// short and would read out of bounds on the last id. The sizes here come from
// the tensors, and the disagreement is asserted so it cannot pass unnoticed.
#pragma once

#include <string>
#include <vector>

#include "vllm/model_executor/model_loader/safetensors_reader.h"
#include "vllm/model_executor/models/gpt2.h"

namespace vllm {
namespace models {
namespace indextts2 {

struct TalkerWeights {
  gpt2::Params params;
  gpt2::Weights backbone;

  // The IndexTTS delta: two vocabularies, two position tables, two heads.
  std::vector<float> text_embedding;      // [text_vocab, H]
  std::vector<float> mel_embedding;       // [mel_codes, H]
  std::vector<float> text_pos_embedding;  // [text_positions, H]
  std::vector<float> mel_pos_embedding;   // [mel_positions, H]
  std::vector<float> final_norm_w, final_norm_b;
  std::vector<float> text_head_w, text_head_b;
  std::vector<float> mel_head_w, mel_head_b;
  std::vector<float> spk_emb_proj_w, spk_emb_proj_b;  // [H, style_dim]
  std::vector<float> lang_embedding;                  // [languages, H]

  int64_t text_vocab = 0;
  int64_t mel_codes = 0;
  int64_t text_positions = 0;
  int64_t mel_positions = 0;
  int64_t languages = 0;
  int64_t style_dim = 0;
};

// `num_attention_heads` is the ONE dimension the checkpoint cannot express:
// GPT-2's fused `c_attn` is [H, 3H] whatever the head count, so nothing in the
// file distinguishes 20 heads from 16. It therefore comes from the config, and
// it is an explicit ARGUMENT rather than a silent default, so a caller cannot
// inherit a number it never chose. `gpt2::Load` owns the rule that it must be
// positive and divide the hidden size; this loader does not restate it.
//
// Throws std::runtime_error naming the missing or misshapen tensor.
TalkerWeights LoadTalker(const SafetensorsFile& file, int64_t num_attention_heads);
TalkerWeights LoadTalker(const std::string& path, int64_t num_attention_heads);

}  // namespace indextts2
}  // namespace models
}  // namespace vllm
