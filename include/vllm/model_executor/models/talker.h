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

}  // namespace talker
}  // namespace models
}  // namespace vllm
