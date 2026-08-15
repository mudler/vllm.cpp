// MiniMax-Music3 — the ACOUSTIC half (W4 + W5 of #672).
//
// Row MODEL-MUSIC-minimax-music3-mini-max-music3-for-conditional-generation,
// .agents/specs/minimax-music3.md phases W4 and W5. Issue #672.
//
// W1 (minimax_music3_loader.h) resolves the six-component diffusers checkpoint,
// folds the vocoder's weight norm and fixes the runtime dtypes. W2/W3
// (minimax_music3_ar.h) produce `frame_hiddens` and the condition mix. This
// header is the rest of the pipeline: the flow-matching DiT that turns a
// condition into Flow-VAE latents, the scheduler that drives it, the window
// bookkeeping the denoise loop performs, and the DAC-style vocoder that turns
// the latents into a 44100 Hz stereo waveform.
//
// ─── THERE IS NO TOKEN GATE ON THIS HALF, AND THAT IS NOT A GAP ─────────────
//
// The acoustic path is a flow-matching denoise loop. It has no logits, no
// vocabulary and no sampler, so no token gate exists to have — spec §0 says so
// and §5 repeats it. (§5's token gate for the AUTOREGRESSIVE half was withdrawn
// separately by W2/W3, for a different reason: upstream has no greedy path
// there either. Two withdrawals, two causes; conflating them is how the wrong
// gate gets re-specified.)
//
// What binds instead is PER-STAGE TENSOR PARITY, each stage against its own
// entry in tests/parity/goldens/minimax_music3_oracle/:
//
//   DiT velocity      denoise_{first,last}_sample_in + condition_chunk0
//                       -> denoise_{first,last}_velocity     (11008 values each)
//   scheduler step    denoise_{first,last}_{sample_in,velocity}
//                       -> denoise_{first,last}_latents_out  (BIT-EXACT)
//   vocoder           vocoder_input_chunk0 -> waveform       (88064 values)
//
// A CORRELATION COEFFICIENT IS NOT A GATE HERE. Pearson is scale-invariant, so
// a uniformly scaled latent passes it while the song is wrong. Every bound in
// the two gate files is on absolute and relative error, and every one of them
// reports how many values it examined.
//
// ─── DTYPE: fp32 IS UPSTREAM'S CHOICE, AND THE ACCUMULATOR IS NOT THE STORE ──
//
// The acoustic half runs FLOAT32 — `MiniMaxMusic3ResolveRuntimeDtypes`'s
// `kBf16ArFp32Acoustic` for both the transformer and the vocoder, which is the
// converter's default (convert_minimax_music3_to_diffusers.py:208-211) and what
// the oracle confirms it ran. That is the opposite polarity from the AR half,
// which is bf16 and rounds at every op boundary (`ArCompute` in
// minimax_music3_ar.h), and the split is a fact about the checkpoint rather
// than a precision preference. So there is no `Compute` parameter here: fp32 is
// the only configuration this half has.
//
// AGENTS.md's "a buffer or GEMM output that names f32 owes a one-line reason"
// is discharged by that paragraph, once, for the whole half.
//
// SEPARATELY, AND IT IS A DIFFERENT AXIS: every reduction below accumulates in
// DOUBLE and stores FLOAT32. That is wider than torch, whose CPU sgemm
// accumulates in float32, and it is deliberate — it is what
// `vocoder1d::Conv1d`, `vocoder1d::ConvTranspose1d` and
// `music3::LinearNoBias` already do, so the accumulator width is the tree's
// established host-reference convention and not a new decision. It costs no
// memory: no buffer, weight or output is stored wider than the fp32 the
// checkpoint carries, which is the thing AGENTS.md's too-wide rule is about.
// The consequence is measured rather than assumed — see the tolerance notes in
// tests/parity/test_minimax_music3_acoustic_real.cpp, where the full-scale
// bounds are calibrated against torch reproducing the goldens AGAINST ITSELF.
//
// ─── THE CONFIGS ARE W1'S, NOT NEW ONES ─────────────────────────────────────
//
// `MiniMaxMusic3TransformerConfig`, `MiniMaxMusic3VocoderConfig` and
// `MiniMaxMusic3SchedulerConfig` come from minimax_music3_loader.h. A second
// copy of `upsampling_ratios` or `rotary_dim` is exactly the keyed duplicate
// that drifts, and the loader's enumeration is what proves the shapes against
// the real checkpoint.
//
// ─── THE VOCODER IS `vocoder1d`, NOT A FORK ─────────────────────────────────
//
// Every convolution, transposed convolution, pad and activation below is a call
// into `vllm::vocoder1d`. W1 recorded the two findings this relies on and they
// are re-stated here because they are load-bearing:
//
//   * `MiniMaxMusic3Snake1d` (minimax_music3_vocoder.py:25-34) is EXACTLY
//     `vocoder1d::SnakeActivation` with a null `beta` and `logscale = false`,
//     down to `kSnakeEps` — `x + (alpha + 1e-9)^-1 * sin^2(alpha * x)`.
//   * Music3 uses PLAIN snake with no up/downsampling, so
//     `vocoder1d::AliasFreeActivation1d` does NOT apply. Routing through it
//     would be a different activation, not a more careful one.
//
// Nothing in `vocoder1d` is modified, so MiniMax-H3's audio VAE and
// IndexTTS-2.5 are byte-identical; tests/scripts/test_vocoder1d_single_home.py
// still sees one home for each symbol.
//
// ─── WHAT IS NOT HERE ───────────────────────────────────────────────────────
//
// The `SpeechEngine` registration, the `vllm_speech_*` ABI and the example HTTP
// server are W6. The quantized arms are W7. This header composes; it does not
// register.
#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include "vllm/model_executor/models/minimax_music3_loader.h"

namespace vllm {
namespace models {
namespace music3 {

// ---------------------------------------------------------------------------
// The flow-matching scheduler (FlowMatchEulerDiscreteScheduler)
// ---------------------------------------------------------------------------

// `np.linspace(1.0, 1.0 / n, n)` — the ramp the denoise loop hands
// `set_timesteps` (denoise.py:154). It is NOT the sigma schedule: the shift and
// the inversion are applied to it afterwards, and reading this as the schedule
// is a plausible off-by-a-transform that still has the right length.
std::vector<double> DenoiseSigmaRamp(int64_t num_inference_steps);

// What `set_timesteps` leaves behind. `sigmas` is ONE LONGER than `timesteps`:
// the terminal value is appended so `step` can read `sigmas[i + 1]`
// (scheduling_flow_match_euler_discrete.py:375,377).
struct FlowMatchSchedule {
  std::vector<float> timesteps;  // [num_inference_steps]
  std::vector<float> sigmas;     // [num_inference_steps + 1]
};

// `set_timesteps(sigmas=ramp)` (scheduling_flow_match_euler_discrete.py:283-382)
// for the branch this model takes: explicit sigmas, no dynamic shifting, no
// `shift_terminal`, no karras/exponential/beta conversion.
//
// THE ORDER OF THE THREE TRANSFORMS IS LOAD-BEARING and each is separately
// visible in the goldens:
//
//   1. shift      s <- shift * s / (1 + (shift - 1) * s)          (:351)
//   2. invert     s <- 1 - s, and the terminal appended is 1.0    (:372-375)
//   3. timesteps  t <- s * num_train_timesteps, AFTER inversion   (:374)
//
// Music3 ships `shift = 1.0`, which makes step 1 the identity — so a schedule
// gate that only ever sees the shipped config cannot tell a wrong shift formula
// from a right one, and cannot tell inversion-before-shift from
// shift-before-inversion. The reduced-dimension goldens carry `shift3` and
// `train1000` variants for exactly that reason.
//
// Throws when the schedule is empty or `use_dynamic_shifting` is set, which
// upstream refuses without a `mu` this pipeline never computes (:309-310).
FlowMatchSchedule FlowMatchSetTimesteps(const std::vector<double>& sigmas,
                                        const MiniMaxMusic3SchedulerConfig& config);

// One Euler step (scheduling_flow_match_euler_discrete.py:497-511) in the
// deterministic branch this model takes (`stochastic_sampling: false`, no
// per-token timesteps):
//
//   prev = sample + (sigmas[i + 1] - sigmas[i]) * velocity
//
// `step_index` is the loop counter, which for this pipeline is also what
// `index_for_timestep` would resolve — the schedule is reset per window and
// walked from 0, so the timestep lookup upstream performs cannot disagree.
//
// Throws when `step_index` has no successor sigma, rather than reading past the
// terminal value.
std::vector<float> FlowMatchStep(const std::vector<float>& sample,
                                 const std::vector<float>& velocity, int64_t step_index,
                                 const FlowMatchSchedule& schedule);

// ---------------------------------------------------------------------------
// Classifier-free guidance
// ---------------------------------------------------------------------------

// The DiT's guidance scale, the `ComponentSpec` default at denoise.py:180. It is
// 1.7 and NOT the AR half's `kArCfgScale` of 1.5; the two stages guide
// independently and sharing one constant would silently retune the song.
inline constexpr double kDitGuidanceScale = 1.7;

// `ClassifierFreeGuidance.forward` (guiders/classifier_free_guidance.py:114-127)
// in the default (non-"original") formulation:
//
//   pred = pred_uncond + scale * (pred_cond - pred_uncond)
//
// Note which row is the base. The ORIGINAL formulation bases on `pred_cond`
// instead, which at scale 1.7 is a different tensor with the same shape; the
// pipeline never sets `use_original_formulation`, so this is the one.
std::vector<float> ClassifierFreeGuidanceMix(const std::vector<float>& conditional,
                                             const std::vector<float>& unconditional,
                                             double guidance_scale);

// ---------------------------------------------------------------------------
// Window bookkeeping — the denoise loop's overlap machinery
// ---------------------------------------------------------------------------

// before_denoise.py:28-29 and denoise.py:39, decoders.py:30-31. Frame counts on
// the AR timeline; latent counts on the Flow-VAE timeline (~3.445 latents per
// frame), and the two are NOT interchangeable.
inline constexpr int64_t kChunkFrames = 200;
inline constexpr int64_t kChunkHop = 100;
inline constexpr int64_t kOverlapLatentLength = 172;
inline constexpr int64_t kCropLeftLatent = 86;
inline constexpr int64_t kCropRightLatent = 344 - 86;

// `chunk_starts` (before_denoise.py:67-70): one window for a short song,
// otherwise `range(0, num_frames - kChunkHop, kChunkHop)`. The stop bound is
// `num_frames - hop`, not `num_frames`, so the last window is a full one rather
// than a stub — and at exactly `kChunkFrames + 1` frames that is the difference
// between two windows and three.
std::vector<int64_t> ChunkStarts(int64_t num_frames);

// The trailing span a window carries into the next (denoise.py:254-256):
// `[len - 2 * kOverlapLatentLength, len - kOverlapLatentLength)`, both ends
// clamped at 0 and `end` clamped up to `start`. For a window shorter than
// `kOverlapLatentLength` the span is EMPTY, which is the case a naive
// subtraction turns into a negative length.
struct WindowCarrySpan {
  int64_t start = 0;
  int64_t end = 0;
  int64_t length() const { return end - start; }
};
WindowCarrySpan ChunkCarrySpan(int64_t latent_length);

// The per-step overlap blend (denoise.py:208-212), in place over the leading
// `overlap` columns of `latents` [channels, length]:
//
//   latents[..., :overlap] = (1 - (1 - 1e-6) * t) * noise_prompt
//                          + t * previous_latent[..., :overlap]
//
// The `(1 - 1e-6)` is invisible at t = 0 and at t = 1 and matters everywhere
// between, which is why the goldens blend at three flow times rather than one.
// `previous_latent` may be WIDER than `overlap`; only its leading columns are
// read, exactly as upstream's slice does.
void BlendOverlap(std::vector<float>& latents, int64_t channels, int64_t length,
                  const std::vector<float>& noise_prompt,
                  const std::vector<float>& previous_latent, int64_t previous_length,
                  int64_t overlap, double time_value);

// The span of one window's decoded waveform that survives the stitch
// (decoders.py:85-87). Every window but the first drops its leading
// `kCropLeftLatent` latent frames; every window but the last drops its trailing
// `kCropRightLatent`. `right_exclusive` is an index, not a count.
struct WaveformCropSpan {
  int64_t left = 0;
  int64_t right_exclusive = 0;
  int64_t length() const { return right_exclusive - left; }
};
WaveformCropSpan VocoderCropSpan(int64_t chunk_index, int64_t num_chunks,
                                 int64_t waveform_length, int64_t hop_length);

// ---------------------------------------------------------------------------
// The flow-matching DiT (MiniMaxMusic3Transformer1DModel)
// ---------------------------------------------------------------------------

// `MiniMaxMusic3RotaryEmbedding.theta` (transformer_minimax_music3.py:45). Not a
// config key: the checkpoint's `rotary_dim` is, the base is not, so it lives
// here rather than in `MiniMaxMusic3TransformerConfig`.
inline constexpr double kDitRotaryTheta = 10000.0;

// `nn.LayerNorm(dim)`'s default eps (transformer_minimax_music3.py:134,136).
// This model's norms are LayerNorms with a BIAS, unlike the RMSNorms of the AR
// half — reading one as the other drops the bias and the mean subtraction.
inline constexpr double kDitLayerNormEps = 1e-5;

struct DitLayerWeights {
  std::vector<float> norm1_weight;  // [inner]
  std::vector<float> norm1_bias;    // [inner]
  std::vector<float> to_q;          // [attn_inner, inner] row-major (out, in)
  std::vector<float> to_k;          // [attn_inner, inner]
  std::vector<float> to_v;          // [attn_inner, inner]
  std::vector<float> to_out;        // [inner, attn_inner]  (`attn.to_out.0`)
  std::vector<float> norm2_weight;  // [inner]
  std::vector<float> norm2_bias;    // [inner]
  std::vector<float> ff_in_weight;  // [2 * ff_inner, inner]
  std::vector<float> ff_in_bias;    // [2 * ff_inner]
  std::vector<float> ff_out_weight; // [inner, ff_inner]
  std::vector<float> ff_out_bias;   // [inner]
};

struct DitWeights {
  std::vector<float> time_proj_weight;            // [fourier_dim / 2, 1]
  std::vector<float> time_embed_linear_1_weight;  // [inner, fourier_dim]
  std::vector<float> time_embed_linear_1_bias;    // [inner]
  std::vector<float> time_embed_linear_2_weight;  // [inner, inner]
  std::vector<float> time_embed_linear_2_bias;    // [inner]
  std::vector<float> preprocess_conv_weight;      // [concat, concat, 1]
  std::vector<float> proj_in_weight;              // [inner, concat]
  std::vector<DitLayerWeights> layers;
  std::vector<float> proj_out_weight;             // [in_channels, inner]
  std::vector<float> postprocess_conv_weight;     // [in_channels, in_channels, 1]
};

// The tensor names the DiT owes, in the order the checkpoint ships them. The
// same walk `EnumerateMiniMaxMusic3TransformerTensors` performs, so W1's
// accounting and W4's consumption cannot drift; returned unsorted, in module
// order, because that is the order this header's structs hold them in.
std::vector<std::string> DitTensorNames(const MiniMaxMusic3TransformerConfig& config);

// Structure a name -> float32 map into `DitWeights`, or THROW naming the tensor
// that is missing or mis-sized.
//
// IT CONSUMES `tensors`: each entry is MOVED out. The shipped DiT is 9.1 GB of
// fp32 and a copy would double that for no reason; the parameter is a non-const
// reference so the cost is visible at the call site rather than hidden.
DitWeights DitWeightsFromTensors(const MiniMaxMusic3TransformerConfig& config,
                                 std::map<std::string, std::vector<float>>& tensors);

// `MiniMaxMusic3FourierEmbedding.forward` (transformer_minimax_music3.py:37-39):
//
//   angles = 2*pi * t * weight^T ;  out = cat(cos(angles), sin(angles))
//
// COS FIRST. `GaussianFourierProjection` in the same file family emits sin
// first, and the two are a permutation of each other that `time_embed` maps to
// entirely different embeddings. Returns [embedding_dim].
std::vector<float> FourierTimeEmbedding(double timestep, const std::vector<float>& weight,
                                        int64_t embedding_dim);

// `TimestepEmbedding` (embeddings.py): linear_1 -> SiLU -> linear_2, both
// biased. Returns [inner_dim].
std::vector<float> DitTimestepEmbedding(const std::vector<float>& fourier,
                                        const MiniMaxMusic3TransformerConfig& config,
                                        const DitWeights& weights);

// The rotary tables (transformer_minimax_music3.py:51-56). Both are
// [seq_len, rotary_dim] — `freqs` is built at HALF the rotary width and then
// concatenated with itself, so the second half of each row repeats the first.
struct DitRotaryTables {
  std::vector<float> cos;  // [seq_len, rotary_dim]
  std::vector<float> sin;  // [seq_len, rotary_dim]
};
DitRotaryTables BuildDitRotaryTables(int64_t seq_len, int64_t rotary_dim,
                                     double theta = kDitRotaryTheta);

// `_apply_partial_rotary_emb` (transformer_minimax_music3.py:59-71), in place
// over [seq_len, heads * head_dim] with the head as the SLOW axis inside a row.
//
// ONLY THE LEADING `rotary_dim` OF EACH HEAD ROTATES; the rest is copied
// through. Music3 ships rotary_dim 32 of head_dim 64, so an implementation that
// rotates the whole head is still finite, still the right shape, and wrong on
// half of every head. The rotate-half split is over the ROTARY window, not over
// the head.
void ApplyPartialRotary(std::vector<float>& x, int64_t seq_len, int64_t heads,
                        int64_t head_dim, const DitRotaryTables& tables);

// The DiT forward (transformer_minimax_music3.py:196-242).
//
//   `latents`   [in_channels, length], CHANNEL-MAJOR
//   `condition` [length, condition_dim], FRAME-MAJOR — the condition mix's own
//               output layout (minimax_music3_ar.h :: ConditionMix), so no
//               transpose happens at the call site
//   returns     [in_channels, length], the flow-matching VELOCITY
//
// Pass an all-zeros `condition` for the unconditional branch (denoise.py:204);
// upstream does not re-encode an empty prompt.
//
// Three shapes inside that a plausible implementation gets wrong:
// the input is `cat(latents, ZEROS_LIKE(latents), condition^T)` — the middle
// block is a genuine zero pad and not a second copy of the latents; the two
// 1x1 convolutions are RESIDUAL (`conv(x) + x`, :220,:238), not replacements;
// and the timestep embedding is prepended as one EXTRA TOKEN that the rotary
// sees and `proj_out` then drops (:227,:236).
std::vector<float> DitForward(const std::vector<float>& latents, int64_t length,
                              const std::vector<float>& condition, double timestep,
                              const MiniMaxMusic3TransformerConfig& config,
                              const DitWeights& weights);

// ---------------------------------------------------------------------------
// The vocoder (MiniMaxMusic3Vocoder), over the shared `vocoder1d` primitives
// ---------------------------------------------------------------------------

struct VocoderResidualUnitWeights {
  std::vector<float> snake1_alpha;  // [dim]
  std::vector<float> conv1_weight;  // [dim, dim, 7]
  std::vector<float> conv1_bias;    // [dim]
  std::vector<float> snake2_alpha;  // [dim]
  std::vector<float> conv2_weight;  // [dim, dim, 1]
  std::vector<float> conv2_bias;    // [dim]
};

struct VocoderBlockWeights {
  std::vector<float> snake1_alpha;    // [input_dim]
  std::vector<float> conv_t1_weight;  // [input_dim, output_dim, 2 * stride]
  std::vector<float> conv_t1_bias;    // [output_dim]
  std::vector<VocoderResidualUnitWeights> res_units;  // dilations 1, 3, 9
};

struct VocoderWeights {
  std::vector<float> dec_in_proj_weight;  // [decoder_input_dim, stream_channels, 1]
  std::vector<float> dec_in_proj_bias;    // [decoder_input_dim]
  std::vector<float> conv_in_weight;      // [hidden, decoder_input_dim, 7]
  std::vector<float> conv_in_bias;        // [hidden]
  std::vector<VocoderBlockWeights> blocks;
  std::vector<float> snake_out_alpha;     // [last_output]
  std::vector<float> conv_out_weight;     // [1, last_output, 7]
  std::vector<float> conv_out_bias;       // [1]
};

// The three residual-unit dilations (minimax_music3_vocoder.py:60-62). The
// dilation changes the PADDING, not the stored shape, so it cannot be read back
// off a checkpoint and has to be pinned here.
inline constexpr int64_t kVocoderResidualDilations[] = {1, 3, 9};
inline constexpr int64_t kVocoderResidualUnits = 3;

// Structure W1's folded vocoder weights, or THROW naming the missing tensor.
// The input is `MiniMaxMusic3LoadVocoderWeights`'s output, in which every
// `weight_g`/`weight_v` pair has ALREADY collapsed to one `<module>.weight`
// through `vocoder1d::MaterializeWeightNorm` — so nothing here can read a
// direction as a weight.
VocoderWeights VocoderWeightsFromLoader(const MiniMaxMusic3VocoderConfig& config,
                                        const MiniMaxMusic3VocoderWeights& loaded);

// `MiniMaxMusic3Snake1d.forward` (minimax_music3_vocoder.py:30-34), in place over
// [channels, length]. A thin, deliberate forward to
// `vocoder1d::SnakeActivation(x, C, T, alpha, nullptr, false)` — Music3's snake
// IS that function (W1's finding), and this exists only so the call site reads
// as the upstream module it mirrors.
void VocoderSnake(std::vector<float>& x, int64_t channels, int64_t length,
                  const std::vector<float>& alpha);

// `MiniMaxMusic3VocoderResidualUnit.forward` (:46-48) over [dim, length].
// `padding = (7 - 1) * dilation // 2` keeps the length; the output is
// `x + conv2(snake2(conv1(snake1(x))))`.
std::vector<float> VocoderResidualUnit(const std::vector<float>& in, int64_t dim,
                                       int64_t length, int64_t dilation,
                                       const VocoderResidualUnitWeights& weights,
                                       int64_t* out_len);

// `MiniMaxMusic3VocoderBlock.forward` (:64-68): snake, transposed convolution at
// `padding = ceil(stride / 2)`, then the three residual units in order.
std::vector<float> VocoderBlock(const std::vector<float>& in, int64_t input_dim,
                                int64_t output_dim, int64_t length, int64_t stride,
                                const VocoderBlockWeights& weights, int64_t* out_len);

// The whole decoder (minimax_music3_vocoder.py:100-115).
//
//   `latents`  [latent_channels, length], CHANNEL-MAJOR
//   returns    [2, length * hop] — the STEREO waveform, `tanh`-bounded
//
// THE STEREO FOLD IS A RESHAPE, NOT A CONVOLUTION (:110, :115). The 128 latent
// channels split into two 64-channel streams, the FIRST 64 becoming the left
// channel and the second the right, and each stream is decoded independently
// by the same weights. Interleaving them instead — the other obvious reading of
// "fold 128 into 2 x 64" — produces a correctly shaped, correctly ranged, wrong
// waveform that no length or dtype check can see.
std::vector<float> VocoderDecode(const std::vector<float>& latents, int64_t length,
                                 const MiniMaxMusic3VocoderConfig& config,
                                 const VocoderWeights& weights, int64_t* out_samples);

}  // namespace music3
}  // namespace models
}  // namespace vllm
