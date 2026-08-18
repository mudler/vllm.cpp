// MiniMax-Music3 — the 8.6B `Qwen3ForCausalLM` half of the AUTOREGRESSIVE stage
// (the remainder of W2 of #672).
//
// Row MODEL-MUSIC-minimax-music3-mini-max-music3-for-conditional-generation,
// .agents/specs/minimax-music3.md phase W2. Issue #672.
//
// W2/W3 (minimax_music3_ar.h) landed everything the autoregressive loop does
// AROUND the language model: the prompt, the unconditional row, the vocabulary
// mask, the guided-logit pipeline, `_sample_top_k` up to its draw, the depth
// decoder and the frame feedback embedding. W6 (minimax_music3_speech.h) landed
// everything downstream of the frame hiddens. What neither had is the LOOP
// ITSELF and the model at its centre, and W6's `Synthesize` refused by name for
// exactly that.
//
// This header is that loop: `MiniMaxMusic3SemanticGenerationStep.__call__`
// (encoders.py:281-353) plus `_generate_depth_codes` (:117-142), driven through
// the LANDED Qwen3 dense path.
//
// ─── WHAT THE LANGUAGE MODEL NEEDED THAT IT DID NOT HAVE ────────────────────
//
// `inputs_embeds`. Upstream calls `language_model.model(inputs_embeds=...)`
// TWICE and never once with `input_ids` (encoders.py:311, :353), because the
// frame feedback `_embed_audio_frame` (:106-115) is a SUM of one language-model
// embedding row and seven depth-decoder rows scaled by `num_codebooks^-0.5` — a
// continuous vector corresponding to no vocabulary entry, which no token id can
// spell. `Qwen3DenseModel::ForwardEmbeds` (qwen3.h) is that entry, added
// additively beside the token-id `Forward` that five registrations share, and
// gated bit-identical against it in tests/vllm/models/test_qwen3_forward.cpp.
//
// ─── THE CONDITION MIX'S EIGHT "LAYERS" ARE NOT TRANSFORMER LAYERS ──────────
//
// Recorded here because it is the reading a fresh implementer will reach for
// and it is wrong. `num_condition_layers: 8` does NOT mean the condition
// encoder mixes eight of the language model's 36 hidden layers, so no
// per-layer-output capture is needed from the Qwen3 stack. The eight rows of a
// `frame_hiddens` entry are `cat(last_hidden, depth_hidden_1..7)`
// (encoders.py:343; `FrameHiddenRow`, minimax_music3_ar.h) — ONE language-model
// hidden state and the SEVEN per-depth-step states of the RVQ decoder. Only the
// LAST hidden state of the language model is ever read, which is why
// `ForwardEmbeds`'s `out_hidden` returns the post-final-norm rows and nothing
// deeper.
//
// ─── THE DRAW IS A PARAMETER, FOR W6's REASON ───────────────────────────────
//
// Spec §5: upstream's autoregressive stage has NO greedy path. `_sample_top_k`
// (encoders.py:94-103) ends in `torch.multinomial` against a seeded
// `torch.Generator`, so `rvq_codes.npy` is a seeded SAMPLE and reproducing it
// would be reproducing torch's Mersenne-Twister rather than this model. A
// second, independent reason survives even a bit-exact RNG: both stages sample
// from a CFG mix whose UNCONDITIONAL row is not in the golden set at all
// (encoders.py:132,343 store `[:1]`).
//
// So the draw is a PARAMETER of the loop, exactly as `Music3NoiseSource` is a
// parameter of `Music3DenoiseChunks` for the same reason: the engine supplies a
// seeded categorical draw, and a gate supplies the capture's OWN codes. That
// TEACHER-FORCES the language model onto the oracle's trajectory, which is the
// only entry at which `frame_hiddens[:, :4096]` is comparable at all.
//
// ─── ONE RECORDED DTYPE DEVIATION, AND ONE MIRRORED ROUNDING ────────────────
//
// The shared Qwen3 dense forward emits f32 logits from an f32-accumulating
// lm_head GEMM; upstream's `language_model.lm_head` is a bf16 `nn.Linear` whose
// output is bf16 and is only then widened by `.float()` (encoders.py:312). The
// SHARED seam is not forked for this — five registrations and a token-exact row
// ride it — so the rounding is mirrored HERE, on the way out, by rounding the
// semantic logit row to bf16 before the guidance pipeline. That is upstream's
// memory format, restored at the one place it is observable, and it is a
// narrowing rather than a widening (.agents/porting.md, "mirror the memory
// format").
#pragma once

#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <vector>

#include "vllm/model_executor/models/minimax_music3_ar.h"
#include "vllm/model_executor/models/minimax_music3_loader.h"
#include "vllm/model_executor/models/qwen3.h"
#include "vllm/tokenizer/tokenizer.h"
#include "vllm/transformers_utils/hf_config.h"
#include "vt/device.h"

namespace vllm {
namespace models {
namespace music3 {

// ---------------------------------------------------------------------------
// The draw
// ---------------------------------------------------------------------------

// Which draw the loop is asking for. Passed so a gate can teacher-force without
// counting calls: the pair (frame_index, codebook) is exactly the position in
// `rvq_codes.npy`, whose row 0 is the priming decode step.
struct Music3Draw {
  // The AR loop's `frame_index` (encoders.py:299). Row 0 of `rvq_codes` is
  // frame_index 0, the priming step that emits no frame.
  int64_t frame_index = 0;
  // 0 for the language model's SEMANTIC draw, 1..num_codebooks-1 for the depth
  // decoder's residual draws — the column index in `rvq_codes`.
  int64_t codebook = 0;
};

// Draws ONE index from a categorical distribution. `probs` sums to 1 and is what
// `TopKProbabilities` returned; the return value indexes `probs`.
//
// For the SEMANTIC draw (`codebook == 0`) that index is a language-model
// VOCABULARY id — so `kAudioEndTokenId` is a legal answer and stops the loop,
// and any other answer must lie in the semantic code window. For a DEPTH draw it
// is a residual code in [0, audio_vocab_size).
using Music3CodeSampler =
    std::function<int64_t(const std::vector<float>& probs, const Music3Draw& draw)>;

// A seeded categorical draw. std::mt19937_64 + std::discrete_distribution:
// deterministic FOR THIS PORT at a given seed, and NOT torch's stream. Named
// rather than implied, because "seeded" reads like "reproducible against the
// oracle" and spec §5 is explicit that it is not.
Music3CodeSampler Music3SeededSampler(int64_t seed);

// ---------------------------------------------------------------------------
// The weights the loop needs
// ---------------------------------------------------------------------------

// The two models the autoregressive stage runs per frame, plus the tokenizer the
// prompt goes through. Held together because a caller that has one without the
// others cannot take a single step.
struct Music3ArWeights {
  // language_model/, through the SHARED dense loader
  // (`LoadQwen3ForCausalLMWeights`) — no Music3-specific weight path exists.
  HfConfig lm_config;
  Qwen3DenseWeights lm;
  MiniMaxMusic3RvqDepthDecoderConfig depth_shipped;
  DepthDecoderConfig depth_config;
  DepthDecoderWeights depth;

  // tokenizer/tokenizer.json. `Qwen2Tokenizer` with no post-processor, so
  // upstream's `tokenizer(text)` is a plain `Encode` (verified: the capture's
  // own prompt encodes to the same 61 ids on both sides).
  std::vector<int32_t> Encode(const std::string& text) const;

  int64_t hidden_size() const { return lm_config.hidden_size; }

  // The embedding table's raw bf16 rows [vocab, hidden]. The forward consumes
  // `lm.embed_tokens` on the DEVICE; `_embed_audio_frame` and
  // `_generate_depth_codes` read individual rows on the HOST
  // (encoders.py:107,127). Same bytes, second reader — not a second copy.
  const uint16_t* embed_rows() const {
    return reinterpret_cast<const uint16_t*>(lm.embed_tokens.bytes.data());
  }
  // One embedding row as host floats carrying bf16-exact values.
  std::vector<float> EmbedRow(int64_t token_id) const;

  // Owned storage. `std::optional` only because `tok::Tokenizer` has no public
  // default constructor (it is built by a named factory); `Encode` refuses on an
  // unloaded one rather than dereferencing.
  std::optional<tok::Tokenizer> tokenizer;
};

// Load `language_model/`, `rvq_depth_decoder/` and `tokenizer/` from a resolved
// diffusers-arm checkpoint, at the RUNTIME dtypes spec §2.1 fixes (bf16 for both
// models — `MiniMaxMusic3ResolveRuntimeDtypes(kBf16ArFp32Acoustic)`).
//
// ~17.2 GB for the language model and ~1.3 GB for the depth decoder, so this is
// the expensive call in the pipeline and it is separated from the loop for that
// reason: a gate loads once and steps many times.
Music3ArWeights Music3LoadArWeights(const MiniMaxMusic3Paths& paths,
                                    const MiniMaxMusic3Config& config);

// ---------------------------------------------------------------------------
// The language model's decode session
// ---------------------------------------------------------------------------

// The two prompt rows and their paged KV, stepped one frame-feedback embedding
// at a time.
//
// TWO ROWS, always: upstream batches the conditional prompt and its
// classifier-free counterpart and guides between them at EVERY step
// (encoders.py:216-217, :313-315). Row 0 is conditional, row 1 unconditional.
// They never interact — attention is per row — so row 0's hidden states are what
// a single-row run would produce, which is why `frame_hiddens` stores only
// `[:1]`.
class Music3LmSession {
 public:
  // `max_positions` bounds the KV allocation: the prompt plus one position per
  // feedback step. Throws when it exceeds the checkpoint's
  // `max_position_embeddings`, which is a context limit on our side too and is
  // ENFORCED rather than discovered (spec §7).
  //
  // The cache lands WHERE THE QUEUE POINTS. `dense_attn::KvSlice` builds its
  // tensor views with `d.q.device` (dense_attn_block.h:233), so a host
  // `std::vector` handed to a CUDA forward is a host pointer wearing a device
  // tensor's label — finite, correctly shaped, and read by a kernel that cannot
  // dereference it. That is why this constructor branches on the queue instead
  // of always allocating the vectors it always allocated.
  Music3LmSession(const Music3ArWeights& weights, vt::Queue& queue, int64_t max_positions);

  // Frees the paged KV when it lives on a DEVICE. The CPU arm's cache is
  // `std::vector` storage and needs no destructor at all; this exists because
  // `vt::Backend::Alloc` has no owner of its own.
  ~Music3LmSession();
  Music3LmSession(const Music3LmSession&) = delete;
  Music3LmSession& operator=(const Music3LmSession&) = delete;

  // `language_model.model(inputs_embeds=embed_tokens(text_ids), use_cache=True)`
  // (encoders.py:310-311). Both rows carry the SAME number of tokens because the
  // unconditional row is a token-for-token rewrite of the conditional one.
  //
  // Fills `out_hidden` with the LAST position's post-final-norm rows [2, H] and
  // `out_logits` with `lm_head` over those same rows [2, vocab] — one forward,
  // because upstream reads both off the one call.
  void Prefill(const std::vector<int32_t>& conditional_ids,
               const std::vector<int32_t>& unconditional_ids,
               std::vector<float>* out_hidden, std::vector<float>* out_logits);

  // One decode step from `_embed_audio_frame`'s output (encoders.py:352-353).
  // `embeds` is [2, H] host float carrying bf16-exact values; both rows are the
  // same vector upstream, and that is not assumed here — each row is taken as
  // given.
  void Step(const std::vector<float>& embeds, std::vector<float>* out_hidden,
            std::vector<float>* out_logits);

  int64_t positions() const { return position_; }
  int64_t rows() const { return kRows; }

 private:
  static constexpr int64_t kRows = 2;

  std::vector<float> RunOne(const std::vector<uint16_t>& embeds_bf16, int64_t tokens_per_row,
                            std::vector<float>* out_hidden);

  const Music3ArWeights& weights_;
  vt::Queue& queue_;
  int64_t block_size_ = 0;
  int64_t blocks_per_row_ = 0;
  int64_t position_ = 0;  // tokens already in the cache, per row
  // Exactly ONE of these carries the cache. `kv_storage_` is the CPU arm's,
  // untouched; `device_kv_` holds one `vt::Backend::Alloc` block per layer on a
  // device arm and is empty on CPU. They are not both populated, and the
  // destructor frees only what the device arm took.
  std::vector<std::vector<uint16_t>> kv_storage_;
  std::vector<void*> device_kv_;
  std::vector<PagedKvCache> attn_kv_;
};

// ---------------------------------------------------------------------------
// The loop
// ---------------------------------------------------------------------------

// What one autoregressive stage produced. Counts are returned rather than
// logged: a stage that cannot say how many frames it emitted has not reported.
struct Music3ArResult {
  // [frames, num_codebooks * hidden_size], the layout `FrameHiddenRow` produces
  // and `Music3DenoiseChunks` consumes.
  std::vector<float> frame_hiddens;
  // [calls, num_codebooks] row-major, in `rvq_codes.npy`'s own layout — column 0
  // is the SEMANTIC code with `kAudioCodeOffset` already subtracted, columns
  // 1..7 the residual codes. Row 0 is the priming step that emits no frame, so
  // `rows[1:]` align with `frame_hiddens`.
  std::vector<int32_t> codes;
  int64_t frames = 0;
  int64_t calls = 0;
  // True when the language model emitted `kAudioEndTokenId` and the loop stopped
  // early (encoders.py:325-326) rather than reaching the frame budget.
  bool stopped_on_end_token = false;
};

// `MiniMaxMusic3SemanticGenerationStep.__call__` (encoders.py:299-353).
//
// `prompt_ids` is the CONDITIONAL row; the unconditional row is derived with
// `UnconditionalPromptIds`. `max_frames` is `MaxArFrames`'s answer, already
// resolved.
//
// THROWS when the stage produces zero frames, mirroring upstream's own
// `ValueError` (encoders.py:349-350) rather than handing the acoustic half an
// empty tensor.
Music3ArResult Music3GenerateFrameHiddens(const std::vector<int32_t>& prompt_ids,
                                          int64_t max_frames,
                                          const Music3ArWeights& weights,
                                          const Music3CodeSampler& sampler,
                                          vt::Queue& queue);

// The same loop with the language model's own hidden states SUPPLIED rather than
// produced — the teacher-forcing entry the LLM parity gate drives, and nothing
// else. Kept beside the real loop instead of duplicated inside a test so the two
// cannot drift: `Music3GenerateFrameHiddens` is this function with a session
// attached.
//
// Not exposed on any request path: a caller with the hidden states already has
// what the loop exists to compute.
std::vector<float> Music3DepthStage(const std::vector<float>& last_hidden_conditional,
                                    const std::vector<float>& last_hidden_unconditional,
                                    int64_t frame_index, const Music3ArWeights& weights,
                                    const Music3CodeSampler& sampler,
                                    std::vector<int32_t>* out_frame_codes);

}  // namespace music3
}  // namespace models
}  // namespace vllm
