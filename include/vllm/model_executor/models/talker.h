// IndexTTS-2.5 talker embedding scaffolding (#634).
//
// The talker (`UnifiedVoice`, model_v2.py) runs a GPT-2 backbone -- already
// ported -- over a sequence assembled from text tokens, mel codes and a speaker
// conditioning latent. This header covers that assembly.
//
// ITS POSITION EMBEDDINGS ARE NOT GPT-2's `wpe`. `LearnedPositionEmbeddings`
// (model_v2.py:244-256) is a SEPARATE learned table, added on top of the token
// embedding before the backbone runs -- so the backbone's own wpe applies as
// well. Treating them as one table silently halves the positional signal.
#pragma once

#include <cstdint>
#include <vector>

#include "vllm/model_executor/models/gpt2.h"

namespace vllm {
namespace models {
namespace talker {

// LearnedPositionEmbeddings::forward — rows 0 .. seq_len-1 of the table, i.e.
// the positions a FULL sequence occupies.
std::vector<float> PositionRows(const std::vector<float>& table, int64_t dim, int64_t seq_len);

// LearnedPositionEmbeddings::get_fixed_embedding — the SINGLE row at `index`.
//
// This is the incremental-decode path: at generation step n the position is n,
// not 0. Returning row 0 每 step makes every generated frame believe it is the
// first, which still decodes to audio and destroys the prosody. The distinction
// cannot be seen from shapes.
std::vector<float> PositionRowAt(const std::vector<float>& table, int64_t dim, int64_t index);

// Token embedding lookup plus the learned position row, the assembly the talker
// performs for both its text and mel streams.
std::vector<float> EmbedWithPositions(const std::vector<int64_t>& tokens,
                                      const std::vector<float>& token_table,
                                      const std::vector<float>& pos_table, int64_t dim,
                                      int64_t vocab_size);


// The talker's PROMPT: what the GPT-2 backbone actually sees before it starts
// emitting mel codes (#634).
//
// Upstream `indextts/gpt/model_v2_5.py:612-677` `prepare_gpt_inputs`, index-tts
// @4f8792ff120cd3ea470dd511e997a17c86cddd10.
//
//   text  = [start_text] + strip(text, start/stop) + [stop_text]
//   emb   = text_embedding[text] + text_pos_embedding[0..n) + lang_embedding[lang]
//   row   = [ zeros(padding) ][ conditioning ][ emb ]
//   mask  = zeros(padding), then ones to target_len + 1
//   ids   = ones(target_len + 1), last element start_mel_token
//
// Four things here are easy to get wrong and all of them still run:
//
//   - THE PAD GOES IN FRONT OF THE CONDITIONING, not between it and the text.
//     Upstream `insert(0, pad)`. Padding between the two shifts the conditioning
//     away from position 0 and leaves the text where it was.
//   - the language embedding is added to EVERY text position, not just the
//     first, and not as a separate token.
//   - start and stop text tokens already present in the input are STRIPPED
//     before the pair is re-added, so passing an already-delimited sequence does
//     not double them.
//   - the attention mask is target_len + 1 long -- the extra slot is the
//     start_mel_token that has not been embedded yet -- and only the PAD is
//     masked, never the conditioning.
struct PromptWeights {
  std::vector<float> text_embedding;      // [text_vocab, dim]
  std::vector<float> text_pos_embedding;  // [text_positions, dim]
  std::vector<float> lang_embedding;      // [languages, dim]
};

struct Prompt {
  std::vector<float> embeds;        // [target_len, dim]
  std::vector<int64_t> input_ids;   // [target_len + 1]
  std::vector<int64_t> attention_mask;  // [target_len + 1]
  int64_t target_len = 0;
};

struct PromptConfig {
  int64_t dim = 0;
  int64_t start_text_token = 0;
  int64_t stop_text_token = 0;
  int64_t start_mel_token = 0;
  // The declared text length the batch is padded to; upstream's `L`.
  int64_t text_slots = 0;
};

// `conditioning` is [cond_rows, dim] -- upstream builds three rows, the speaker
// projection plus the emotion vector followed by two zero rows.
Prompt PrepareInputs(const PromptConfig& cfg, const PromptWeights& w,
                     const std::vector<float>& conditioning, int64_t cond_rows,
                     const std::vector<int64_t>& text_tokens, int64_t lang);


// The talker's AUTOREGRESSIVE LOOP: prompt in, mel codes out (#634).
//
// Upstream drives `GPT2InferenceModel.generate` over the prompt
// `PrepareInputs` builds; this is the greedy reference for it. The head is
// `Sequential(final_norm, mel_head)` (`model_v2_5.py:445-446`) -- a LayerNorm
// and then a Linear over `number_mel_codes`.
//
// Greedy only, deliberately. Upstream samples, and the spec already records
// that a token-exact e2e gate is therefore unavailable and not claimed; a
// deterministic reference is what CAN be gated, and it is what any sampling
// implementation must reduce to at temperature zero.
//
// Each accepted code re-enters as `mel_embedding[code] + mel_pos_embedding[step]`,
// indexed by STEP -- its position within the generated sequence -- not by its
// absolute position in the prompt. The two differ by the whole prompt length,
// and using the absolute one reads real rows from a real table.
struct GenerateWeights {
  std::vector<float> mel_embedding;      // [mel_codes, dim]
  std::vector<float> mel_pos_embedding;  // [mel_positions, dim]
  std::vector<float> final_norm_w, final_norm_b;
  std::vector<float> mel_head_w;  // [mel_codes, dim]
  std::vector<float> mel_head_b;  // [mel_codes]
};

struct GenerateConfig {
  int64_t dim = 0;
  int64_t mel_codes = 0;
  int64_t start_mel_token = 0;
  int64_t stop_mel_token = 0;
  int64_t max_mel_tokens = 0;
  double layer_norm_eps = 1e-5;
};

// `prompt_embeds` is [prompt_len, dim] from `PrepareInputs`. Returns the codes
// emitted BEFORE the stop token, which is not included.
std::vector<int64_t> GenerateMelCodes(const GenerateConfig& cfg, const GenerateWeights& w,
                                      const gpt2::Params& params,
                                      const gpt2::Weights& backbone,
                                      const std::vector<float>& prompt_embeds,
                                      int64_t prompt_len);

}  // namespace talker
}  // namespace models
}  // namespace vllm
