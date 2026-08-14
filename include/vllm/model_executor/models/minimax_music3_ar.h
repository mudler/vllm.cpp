// MiniMax-Music3 — the AUTOREGRESSIVE half (W2 + W3 of #672).
//
// Row MODEL-MUSIC-minimax-music3-mini-max-music3-for-conditional-generation,
// .agents/specs/minimax-music3.md phases W2 and W3. Issue #672.
//
// W1 (minimax_music3_loader.h) resolves the six-component diffusers checkpoint
// and materializes its tensors. This header is what CONSUMES three of those
// components: the prompt the `language_model` is driven with, the semantic
// stage's classifier-free-guidance logit pipeline, the learned 8-layer condition
// mix, and the 4-layer RVQ depth decoder.
//
// ─── THE SHAPE OF THE AUTOREGRESSIVE STAGE ──────────────────────────────────
//
// encoders.py:281-353. Per frame, TWO models run:
//
//   1. the GLOBAL language model (a stock `Qwen3ForCausalLM` at vocab 200000)
//      emits one SEMANTIC code, under classifier-free guidance between a
//      conditional and an unconditional prompt row;
//   2. the RVQ DEPTH DECODER expands that one code into the SEVEN residual
//      codebooks, one at a time, and its per-step hidden states — concatenated
//      with the language model's own — are the `frame_hiddens` the acoustic
//      half is conditioned on.
//
// Then `_embed_audio_frame` folds the whole 8-codebook frame back into ONE
// language-model input embedding and the loop advances.
//
// ─── WHAT IS A GATE HERE AND WHAT IS NOT — READ THIS BEFORE TRUSTING §5 ─────
//
// The spec's §5 says the LLM half is "token-exact": "Greedy decode of the code
// sequence is compared against the oracle token-for-token." MEASURED AGAINST
// THE ARTIFACT, THAT IS NOT AVAILABLE, and the artifact wins.
//
// Upstream's autoregressive stage HAS NO GREEDY PATH. `_sample_top_k`
// (encoders.py:94-103) is the only sampler either stage uses, `_AR_SAMPLING_TOP_K`
// is a module constant of 50 with no temperature and no argmax branch, and the
// final step is `torch.multinomial(probs, 1, generator=generator)`. So the
// committed `rvq_codes.npy` is a SEEDED SAMPLE, not an argmax, and reproducing
// it token-for-token means reproducing torch's CPU Mersenne-Twister and its
// multinomial, not reproducing this model.
//
// A second, independent reason the same conclusion holds: BOTH stages sample
// from a CFG mix of a conditional and an UNCONDITIONAL row (encoders.py:327-328,
// :134-135), and `frame_hiddens` stores the conditional row only
// (encoders.py:343 `last_hidden[:1]`, :132 `hidden[:1]`). The unconditional
// branch is not in the golden set at all, so even with a bit-exact RNG the codes
// could not be re-derived from what is committed.
//
// What IS therefore gated, and what this header is built around:
//
//   * every DETERMINISTIC stage, against the reduced-dimension float32 goldens
//     produced by executing upstream's own classes
//     (scripts/gen-minimax-music3-ar-goldens.py);
//   * the depth decoder at FULL SCALE with REAL weights, driven by the golden
//     `last_hidden` and the golden codes, against the golden hidden states —
//     which is 25 x 7 x 4096 = 716800 real numbers and cannot be passed by an
//     implementation that has any of the algebra wrong;
//   * the condition mix at FULL SCALE against `condition_chunk0.npy`.
//
// The codes are consumed as INPUTS by those gates rather than predicted. That is
// a weaker claim than §5 makes and it is the true one; the spec is corrected in
// its `## Now`, not worked around here.
//
// ─── DTYPE IS A PARAMETER, BECAUSE UPSTREAM'S IS ────────────────────────────
//
// The gated configuration runs all three AR components in BF16 (spec §2.1 as
// corrected by the oracle; `MiniMaxMusic3ResolveRuntimeDtypes`). A bf16 torch
// module rounds at EVERY op boundary, so an fp32 host reference is not "the same
// computation more precisely" — it is a different one, and over the depth
// decoder's four layers the two diverge by ~1.3 bf16 ULP on average. MEASURED:
// an fp32 forward against the committed bf16 goldens leaves 448450 of 716800
// values beyond one ULP with a mean absolute error of 2.65e-03; mirroring the
// rounding brings the same forward to the numbers the gate now records.
//
// This is the "do not inherit a WIDER dtype" rule of AGENTS.md in its awkward
// direction: nothing here is numerically wrong at fp32, so a token gate could
// never see it, and only a tensor gate against the real weights can.
//
// So `ArCompute` is threaded through every function that has an op boundary.
// `kFloat32` is the default because that is what the reduced-dimension goldens
// are captured in; `kBFloat16` is what the SHIPPED checkpoint runs and what the
// full-scale gate uses. Neither is a widening of a stored value: the caller
// converts the checkpoint's tensors once (`MiniMaxMusic3LoadComponent`), and at
// `kBFloat16` it owes them at bf16 — including the condition encoder, whose FILE
// is fp32 while its RUNTIME is bf16.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace vllm {
namespace models {
namespace music3 {

// Which dtype the AR half's op boundaries round to. See the header note: this
// mirrors a real property of the run, not a precision preference.
enum class ArCompute {
  // The reduced-dimension goldens' dtype. Every accumulation is a double and
  // each result is rounded once to float32.
  kFloat32,
  // The SHIPPED configuration. Accumulation is still a double — torch's bf16
  // matmul accumulates in float32 and its RMSNorm variance in float32 — but the
  // RESULT of each op is rounded to bf16, which is what torch stores.
  kBFloat16,
};

// ---------------------------------------------------------------------------
// The checkpoint contract's constants (encoders.py:38-48)
// ---------------------------------------------------------------------------

// The three tokens the prompt template and the CFG rewrite name explicitly.
inline constexpr int32_t kAudioEndTokenId = 151670;
inline constexpr int32_t kAudioCfgTokenId = 151654;
inline constexpr int32_t kAudioCodeOffset = 151675;
// The semantic codebook occupies [kAudioCodeOffset, +kSemanticVocabSize) of the
// language model's 200000-entry vocabulary; everything else is masked out.
inline constexpr int64_t kSemanticVocabSize = 16384;

inline constexpr int64_t kMaxPromptTokens = 5000;
inline constexpr int64_t kMaxAudioFrames = 9000;

// The reference inference recipe's fixed sampling parameters (encoders.py:46-48).
// `kArCfgScale` applies to BOTH stages; `kArCfgTopK` restricts the semantic
// stage's guided distribution to the CONDITIONAL row's top candidates before
// `kArSamplingTopK` filters the guided one.
inline constexpr double kArCfgScale = 1.5;
inline constexpr int64_t kArCfgTopK = 50;
inline constexpr int64_t kArSamplingTopK = 50;

// ---------------------------------------------------------------------------
// Prompt assembly (encoders.py:54-91, :207-218)
// ---------------------------------------------------------------------------

// The literal template pieces. Upstream's own comment (encoders.py:32-33) is
// that "even whitespace-level changes to the assembled prompt change the
// generated audio", which is why these are constants and not formatting.
inline constexpr const char* kImStart = "<|im_start|>";
inline constexpr const char* kImEnd = "<|im_end|>";
inline constexpr const char* kCaptionStart = "<|caption_start|>";
inline constexpr const char* kCaptionEnd = "<|caption_end|>";
inline constexpr const char* kLyricsStart = "<|lyrics_start|>";
inline constexpr const char* kLyricsEnd = "<|lyrics_end|>";
inline constexpr const char* kAudioStart = "<|audio_start|>";

// `_clean_caption` (encoders.py:54-77). Rewrites `<|k v|>` to "k is v", strips
// the markdown forms the input contract accepts (ATX headings, list bullets,
// bold, italic, horizontal rules), and collapses blank runs.
std::string CleanCaption(const std::string& caption);

// `_normalize_lyrics` (encoders.py:80-91). Keeps only the LEADING structural
// tags of a line and drops any text sharing that line, then splits on "] " and
// " [" and " ^ ", lower-cases every `[Tag]`, and prefixes "[start]\n".
//
// THE REPLACEMENT ORDER IS LOAD-BEARING: "] " runs before " [", so
// "tail [outro] ^ after caret" becomes "tail\n[outro]\n^ after caret" and the
// " ^ " rule then matches nothing. Swapping the two produces a different,
// entirely plausible prompt.
std::string NormalizeLyrics(const std::string& lyrics);

// The assembled prompt string (encoders.py:207-210).
std::string AssembleArPrompt(const std::string& prompt, const std::string& lyrics);

// The unconditional row (encoders.py:216-217): a copy with every token but the
// FIRST and the last TWO replaced by `kAudioCfgTokenId`. Throws when the ids are
// too short for that slice to be well defined.
std::vector<int32_t> UnconditionalPromptIds(const std::vector<int32_t>& ids);

// `min(int(audio_duration * frame_rate), kMaxAudioFrames)` (encoders.py:287).
// Throws when the duration is not positive or rounds to zero frames, mirroring
// upstream's two `ValueError`s rather than generating silence.
int64_t MaxArFrames(double audio_duration_s, double frame_rate);

// ---------------------------------------------------------------------------
// The semantic stage's logit pipeline (encoders.py:318-334)
// ---------------------------------------------------------------------------

// The vocabulary mask, TRUE where a token is BLOCKED. Only the semantic code
// window and the audio-end token survive (encoders.py:318-320).
std::vector<bool> SemanticVocabMask(int64_t vocab_size, int64_t code_offset,
                                    int64_t semantic_vocab_size, int32_t end_token_id);

// The guided semantic logits. `conditional` and `unconditional` are the two raw
// lm_head rows; the mask is applied to BOTH before guidance, the guided row is
// then restricted to the CONDITIONAL row's top `cfg_top_k` and re-masked.
//
// THE RE-MASK IS NOT REDUNDANT (encoders.py:329-333): guidance on two -inf
// logits is (-inf) + (-inf - -inf) * 1.5 = NaN, so without the second mask every
// blocked position becomes a NaN that `_sample_top_k`'s `nan_to_num` would turn
// into a finite -1e9 candidate rather than an impossible one.
std::vector<float> GuidedSemanticLogits(const std::vector<float>& conditional,
                                        const std::vector<float>& unconditional,
                                        const std::vector<bool>& blocked,
                                        int64_t cfg_top_k, double cfg_scale);

// The guided DEPTH logits (encoders.py:134-135): the same mix with no vocabulary
// mask and no top-k pre-restriction, because the residual codebooks span their
// whole `audio_vocab_size`.
std::vector<float> GuidedDepthLogits(const std::vector<float>& conditional,
                                     const std::vector<float>& unconditional,
                                     double cfg_scale);

// `_sample_top_k` MINUS its final `torch.multinomial` (encoders.py:94-100): the
// sanitize / top-k mask / softmax / renormalize that produce the categorical
// distribution. Returns probabilities summing to 1.
//
// The draw itself is deliberately NOT here. It is `torch.multinomial` against a
// seeded `torch.Generator`, so reproducing it is reproducing torch's RNG; see
// the header note on why no gate in this port claims to.
std::vector<float> TopKProbabilities(const std::vector<float>& logits, int64_t top_k);

// ---------------------------------------------------------------------------
// The condition mix (condition_embedder_minimax_music3.py:48-76)
// ---------------------------------------------------------------------------

struct ConditionMixConfig {
  int64_t condition_hidden_dim = 4096;
  int64_t num_condition_layers = 8;
  int64_t out_dim = 2048;
  int64_t input_sampling_rate = 24000;
  int64_t input_hop_length = 960;
  int64_t output_sampling_rate = 44100;
  int64_t output_hop_length = 512;
};

struct ConditionMixWeights {
  std::vector<float> layer_weight_logits;  // [num_condition_layers]
  std::vector<float> layer_scale;          // [1]
  std::vector<float> proj_weight;          // [out_dim, condition_hidden_dim, 3]
  std::vector<float> proj_bias;            // [out_dim]
};

// The latent timeline's length (condition_embedder_minimax_music3.py:65-74).
// `max(1, int(frames * out_rate / in_rate * in_hop / out_hop))`, and the
// intermediate is a DOUBLE that is truncated once at the end — computing it as
// integer ratios rounds 86.13 to 86 for the wrong reason and to 87 for the
// wrong inputs.
int64_t ConditionLatentLength(int64_t num_frames, const ConditionMixConfig& config);

// `hidden_states` is [frames, num_condition_layers * condition_hidden_dim] in
// LAYER-MAJOR order (the transpose+reshape of :59-60 makes the layer the SLOW
// axis). Returns [latent_length, out_dim].
//
// Three steps that each have a plausible wrong form: a softmax over the layer
// logits (not a normalize, not a plain weighting), ONE scalar `layer_scale`
// applied to the mixed result (not per layer), and a k=3 p=1 Conv1d over TIME
// (not a pointwise projection). Then NEAREST interpolation to the latent rate.
std::vector<float> ConditionMix(const std::vector<float>& hidden_states,
                                int64_t num_frames, const ConditionMixConfig& config,
                                const ConditionMixWeights& weights,
                                ArCompute compute = ArCompute::kFloat32);

// The softmax over the layer logits, exposed so a gate can see the mix weights
// rather than only their effect.
std::vector<float> ConditionLayerWeights(const std::vector<float>& layer_weight_logits,
                                         ArCompute compute = ArCompute::kFloat32);

// `F.interpolate(..., mode="nearest")` along the last axis, for [channels, in_len]
// row-major input. Upstream's source index is `floor(dst * in_len / out_len)`
// clamped to `in_len - 1` — the ratio is INPUT over OUTPUT, and inverting it is
// a silent off-by-a-scale that still returns the right shape.
std::vector<float> NearestInterpolate1d(const std::vector<float>& in, int64_t channels,
                                        int64_t in_len, int64_t out_len);

// ---------------------------------------------------------------------------
// The RVQ depth decoder (minimax_music3_rvq_depth_decoder.py:91-142)
// ---------------------------------------------------------------------------

struct DepthDecoderConfig {
  int64_t hidden_size = 4096;
  int64_t num_layers = 4;
  int64_t num_attention_heads = 16;
  int64_t intermediate_size = 6144;
  int64_t audio_vocab_size = 1024;
  int64_t num_codebooks = 8;
  int64_t max_position_embeddings = 16;

  int64_t head_dim() const { return hidden_size / num_attention_heads; }
  int64_t residual_codebooks() const { return num_codebooks - 1; }
};

struct DepthDecoderLayerWeights {
  std::vector<float> input_layernorm;           // [H]
  std::vector<float> post_attention_layernorm;  // [H]
  std::vector<float> to_q;                      // [H, H] row-major (out, in)
  std::vector<float> to_k;
  std::vector<float> to_v;
  std::vector<float> to_out;
  std::vector<float> gate_proj;  // [I, H]
  std::vector<float> up_proj;    // [I, H]
  std::vector<float> down_proj;  // [H, I]
};

struct DepthDecoderWeights {
  // [audio_vocab_size * (num_codebooks - 1), H] — the RESIDUAL codebooks only.
  // The semantic codebook is embedded by the LANGUAGE MODEL, not here
  // (encoders.py:126), which is the asymmetry the table's first dimension
  // records.
  std::vector<float> audio_embeddings;
  std::vector<float> projection;     // [H, H]
  std::vector<float> pos_embedding;  // [max_position_embeddings, H]
  std::vector<float> norm;           // [H]
  std::vector<DepthDecoderLayerWeights> layers;
  // One [audio_vocab_size, H] head per RESIDUAL codebook.
  std::vector<std::vector<float>> audio_heads;
};

// RMSNorm with `eps = 1e-6` and elementwise affine, over the last axis of
// [rows, dim] (minimax_music3_rvq_depth_decoder.py:78). Accumulates in double.
// TWO roundings, not one, and that is upstream's shape rather than ours:
// `x * rsqrt(var + eps)` promotes bf16 to float32 (normalization.py:600-601),
// the result is cast BACK to the weight's dtype (:605), and only then is the
// affine weight applied (:606). Collapsing them to one rounding is a different
// number.
std::vector<float> RmsNorm(const std::vector<float>& x, int64_t rows, int64_t dim,
                           const std::vector<float>& weight, double eps = 1e-6,
                           ArCompute compute = ArCompute::kFloat32);

// `y = x @ W^T` for row-major x [rows, in_dim] and W [out_dim, in_dim] — the
// torch `nn.Linear` layout, bias-free everywhere in this decoder.
std::vector<float> LinearNoBias(const std::vector<float>& x, int64_t rows, int64_t in_dim,
                                const std::vector<float>& weight, int64_t out_dim,
                                ArCompute compute = ArCompute::kFloat32);

// The decoder forward (`MiniMaxMusic3RVQDepthDecoder.forward`, :127-142).
// `inputs_embeds` is [seq_len, hidden_size]; returns the post-`norm` hidden
// states at the SAME shape.
//
// Attention is CAUSAL and there is no RoPE and no KV cache: position is carried
// entirely by a learned `pos_embedding` added to the input. Causality is why one
// forward over the whole depth sequence equals upstream's incremental schedule
// step for step — verified bit-exactly against the committed goldens.
//
// Throws when `seq_len > max_position_embeddings`: `pos_embedding` has no row to
// return and upstream would index out of bounds.
std::vector<float> DepthDecoderForward(const std::vector<float>& inputs_embeds,
                                       int64_t seq_len, const DepthDecoderConfig& config,
                                       const DepthDecoderWeights& weights,
                                       ArCompute compute = ArCompute::kFloat32);

// The depth SEQUENCE `_generate_depth_codes` assembles (encoders.py:125-141):
//
//   [ projection(last_hidden),
//     projection(semantic_embed),
//     projection(audio_embeddings(c_i + (i-1) * audio_vocab_size)) for i in 1..n ]
//
// `semantic_embed` is the LANGUAGE MODEL's embedding row for
// `semantic_code + kAudioCodeOffset`, which the caller supplies because this
// component does not own that table. `residual_codes` are c1.., each < the audio
// vocabulary; the OFFSET is what makes one shared table hold seven codebooks,
// and dropping it reads codebook 1's rows for every step.
//
// Returns [2 + residual_codes.size(), hidden_size].
std::vector<float> DepthSequenceEmbeds(const std::vector<float>& last_hidden,
                                       const std::vector<float>& semantic_embed,
                                       const std::vector<int32_t>& residual_codes,
                                       const DepthDecoderConfig& config,
                                       const DepthDecoderWeights& weights,
                                       ArCompute compute = ArCompute::kFloat32);

// `audio_heads[head_index](hidden)` for one [hidden_size] state. `head_index` is
// ZERO-based over the residual codebooks, so codebook c_i uses head i-1
// (encoders.py:133).
std::vector<float> AudioHeadLogits(const std::vector<float>& hidden, int64_t head_index,
                                   const DepthDecoderConfig& config,
                                   const DepthDecoderWeights& weights,
                                   ArCompute compute = ArCompute::kFloat32);

// The per-frame conditioning row the acoustic half consumes (encoders.py:343):
// `cat(last_hidden, depth_hidden_1..depth_hidden_n)`, i.e. the language model's
// hidden state followed by the depth decoder's states at depth steps 1..n.
// Returns [num_codebooks * hidden_size].
std::vector<float> FrameHiddenRow(const std::vector<float>& last_hidden,
                                  const std::vector<float>& depth_hidden_states,
                                  int64_t seq_len, const DepthDecoderConfig& config);

// `_embed_audio_frame` (encoders.py:106-115): the language model's next input
// embedding for a complete frame.
//
//   (lm_embed(semantic + kAudioCodeOffset)
//      + sum_j audio_embeddings(c_j + j * audio_vocab_size)) * num_codebooks^-0.5
//
// The `j * audio_vocab_size` here is a ZERO-based offset over the residual
// codes, unlike `DepthSequenceEmbeds`'s `(i-1) *` on a one-based index — they
// are the same offsets written from different loop bases, and reading one for
// the other shifts every codebook by 1024 rows.
std::vector<float> EmbedAudioFrame(const std::vector<float>& lm_semantic_embed,
                                   const std::vector<int32_t>& residual_codes,
                                   const DepthDecoderConfig& config,
                                   const DepthDecoderWeights& weights,
                                   ArCompute compute = ArCompute::kFloat32);

}  // namespace music3
}  // namespace models
}  // namespace vllm
