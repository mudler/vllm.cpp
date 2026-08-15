// MiniMax-Music3 — the SPEECH-FAMILY registration and the pipeline that joins
// the two halves (W6 of #672).
//
// Row MODEL-MUSIC-minimax-music3-mini-max-music3-for-conditional-generation,
// .agents/specs/minimax-music3.md phase W6. Issue #672.
//
// W1 (minimax_music3_loader.h) resolves the six-component diffusers checkpoint.
// W2/W3 (minimax_music3_ar.h) own the autoregressive half's prompt, guidance and
// condition mix. W4/W5 (minimax_music3_acoustic.h) own the flow-matching DiT,
// the scheduler and the DAC vocoder. Every one of those is a PIECE; nothing
// composed them, and nothing was reachable from outside the tree.
//
// This header is the composition and the seam:
//
//   modular_pipelines/minimax_music3/before_denoise.py -> Music3ChunkPlan
//   modular_pipelines/minimax_music3/denoise.py        -> Music3DenoiseChunks
//   modular_pipelines/minimax_music3/decoders.py       -> Music3DecodeChunks
//   the whole pipeline behind multimodal::SpeechEngine -> Music3SpeechEngine
//
// ─── WHY IT IS A `SpeechEngine` AND NOT A NEW SEAM ──────────────────────────
//
// AGENTS.md: a capability not reachable through the shared surface is not done,
// and a seam is EXTENDED rather than forked. `multimodal::SpeechEngine` already
// carries `channels` and already documents `sample_rate` as the family's NATIVE
// rate rather than a resampled one — which is exactly spec §1.1's 44100 stereo,
// and exactly why SGLang-Omni's 32 kHz stays a caller-side concern.
//
// The one genuine gap was `SpeechGenParams`, which carried a single `text`
// because IndexTTS-2 synthesizes one utterance. It is EXTENDED ADDITIVELY with
// `lyrics`, `description` and three generation controls (speech_engine.h), all
// inert at their defaults. IndexTTS-2.5 is untouched.
//
// ─── WHAT `Synthesize` CANNOT DO YET, SAID HERE RATHER THAN DISCOVERED ──────
//
// The 8.6B `Qwen3ForCausalLM` forward is W2's recorded remainder (spec §5,
// "still owed on the LLM half"): reproducing `frame_hiddens[:, :4096]` means
// running that model teacher-forced through our landed Qwen3 path, which needs
// an `inputs_embeds` entry it does not have. Until that lands the AR HEAD IS
// REFUSED BY NAME — `Music3SpeechEngine::Synthesize` names the missing piece,
// the phase that owes it and the issue, rather than returning silence or a
// waveform produced by something else. Everything downstream of the frame
// hiddens is implemented and gated here.
//
// ─── NO REQUEST'S WAVEFORM CAN EVER EQUAL THE ORACLE'S, AND THAT IS A FACT ──
//
// TWICE OVER, for two independent reasons, and both are structural:
//
//   1. the AR half draws each code with `torch.multinomial` against a seeded
//      `torch.Generator` (encoders.py:94-103) — spec §5 withdrew the token gate
//      for exactly this reason;
//   2. the denoise loop's INITIAL LATENTS are `randn_tensor(...)` against the
//      same generator (denoise.py:117-121).
//
// So "run the pipeline from a prompt and compare the WAV against waveform.npy"
// is not a gate that exists to be written. What IS comparable is the pipeline
// driven from the oracle's OWN recorded inputs, which is why `Music3NoiseSource`
// is a parameter of `Music3DenoiseChunks` rather than a private detail: the
// engine supplies a seeded normal draw, and the gate supplies the capture's
// `denoise_first_sample_in`. That is the only entry at which this loop is
// comparable to the oracle at all, and hiding it would have made the e2e gate
// impossible rather than merely inconvenient.
#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "vllm/model_executor/models/minimax_music3_acoustic.h"
#include "vllm/model_executor/models/minimax_music3_ar.h"
#include "vllm/model_executor/models/minimax_music3_loader.h"
#include "vllm/multimodal/speech_engine.h"

namespace vllm {
namespace models {
namespace music3 {

// The stable registry name. One spelling, used by the registration, by the
// engine's `family()`, and by the server's refusals.
inline constexpr const char* kMusic3SpeechFamily = "minimax-music3";

// spec §1.1: the vocoder's native rate and channel count, resample-free.
inline constexpr int64_t kMusic3SampleRate = 44100;
inline constexpr int64_t kMusic3Channels = 2;

// Upstream's own InputParam defaults, mirrored rather than re-chosen:
// `audio_duration` 60.0 (encoders.py:253) and `num_inference_steps` 30
// (denoise.py:144, :190). `kDitGuidanceScale` (1.7, denoise.py:180) lives in
// minimax_music3_acoustic.h and is the DiT's; the AR half's 1.5 is a module
// constant of the reference recipe (encoders.py:46-48) and upstream exposes no
// knob for it, so neither does this.
inline constexpr double kMusic3DefaultDurationSeconds = 60.0;
inline constexpr int64_t kMusic3DefaultInferenceSteps = 30;

// The autoregressive frame rate, `input_sampling_rate / input_hop_length`
// (modular_pipeline.py:39-44). 24000 / 960 = 25 Hz for the shipped checkpoint,
// and it is READ from the condition-encoder config rather than assumed.
double Music3FrameRate(const MiniMaxMusic3ConditionEncoderConfig& config);

// ---------------------------------------------------------------------------
// Detection
// ---------------------------------------------------------------------------

// Does this directory hold the DIFFUSERS arm of MiniMax-Music3?
//
// It INSPECTS the artifact rather than trusting the path spelling, which is
// chosen by whoever repackaged the checkpoint: `modular_model_index.json` must
// name this pipeline class AND every component directory W1 resolves must
// exist. NEVER THROWS — a detector's bad day must not deny every other family a
// chance to claim the checkpoint (speech_engine.h).
//
// The NATIVE arm (`qwen_7B/` + `flowmatching_vae.pth` + `dav.pth`) is NOT
// claimed here: it holds every weight this port needs in a layout nothing reads,
// and claiming it would turn W1's by-name refusal into a load failure whose
// message came from a different layer. `MiniMaxMusic3IsNativeArm` is what says
// which it is.
bool Music3DetectCheckpoint(const std::string& path);

// ---------------------------------------------------------------------------
// The chunk plan (before_denoise.py) — pure, so it gates without a checkpoint
// ---------------------------------------------------------------------------

// One denoise window: which AR frames it covers, and its latent length.
struct Music3Chunk {
  int64_t frame_start = 0;
  int64_t frame_end = 0;    // exclusive, clamped at num_frames
  int64_t latent_length = 0;

  int64_t frames() const { return frame_end - frame_start; }
};

// `chunk_starts` + each window's clamped end and latent length
// (before_denoise.py:67-70, denoise.py:79-80). The END IS CLAMPED at
// `num_frames`, so the last window is usually SHORTER than kChunkFrames while
// still starting on the hop grid — reading the length as always 200 gives a
// window that runs off the end of the hidden states.
std::vector<Music3Chunk> Music3ChunkPlan(int64_t num_frames,
                                         const ConditionMixConfig& config);

// ---------------------------------------------------------------------------
// The denoise loop (denoise.py) and the decode (decoders.py)
// ---------------------------------------------------------------------------

// The window's initial latents, [in_channels, latent_length], channel-major.
// `chunk_index` is passed so a source can be per-window without keeping state.
//
// This is a PARAMETER for the reason the header note gives: upstream draws it
// from a seeded `torch.Generator` we do not reproduce, so it is the one place at
// which the loop can be pinned to the oracle's own trajectory.
using Music3NoiseSource =
    std::function<std::vector<float>(int64_t channels, int64_t length, int64_t chunk_index)>;

// A seeded standard-normal source. std::mt19937_64 + std::normal_distribution:
// deterministic FOR THIS PORT at a given seed, and NOT torch's stream. Named
// rather than implied, because "seeded" reads like "reproducible against the
// oracle" and it is not.
Music3NoiseSource Music3SeededNoise(int64_t seed);

// Everything the loop needs that is not a per-request knob.
struct Music3AcousticWeights {
  ConditionMixWeights condition;
  DitWeights dit;
  VocoderWeights vocoder;
};

// Load `condition_encoder/`, `transformer/` and `vocoder/` from a resolved
// checkpoint, at spec §2.1's runtime dtypes: the condition mix bf16 (its FILE is
// fp32 — the AR half's dtype is what it runs in), the DiT and the vocoder fp32,
// with every `weight_g`/`weight_v` pair folded.
//
// ~10 GB, and staged SEPARATELY from the AR half's ~18.5 GB on purpose: upstream
// drives its own offload hooks between the two stages (encoders.py:302-309) and
// nothing needs both resident at once, since `frame_hiddens` is the only thing
// that crosses.
Music3AcousticWeights Music3LoadAcousticWeights(const MiniMaxMusic3Paths& paths,
                                                const MiniMaxMusic3Config& config);

struct Music3DenoiseOptions {
  int64_t num_inference_steps = kMusic3DefaultInferenceSteps;
  double guidance_scale = kDitGuidanceScale;
};

// The whole window loop (denoise.py:196-260 + the chunk condition/prepare/carry
// steps around it). `frame_hiddens` is [num_frames, num_codebooks * hidden] in
// the layout `FrameHiddenRow` produces. Returns one latent tensor per window,
// each [in_channels, latent_length], UNCROPPED — the crop is the decode's
// (decoders.py:85-87), and cropping twice is silent truncation.
//
// Four things inside that a plausible implementation gets wrong, each of them
// upstream's own order rather than a preference:
//   * the previous window's CONDITION is spliced over the overlap BEFORE the
//     noise is drawn (denoise.py:85-88), so the noise length follows the spliced
//     condition rather than the raw one;
//   * the overlap is BLENDED toward the previous latents at EVERY step
//     (denoise.py:208-212) and then RESTORED exactly after the last one
//     (denoise.py:249-250) — the blend is not a substitute for the restore;
//   * the carry span is taken from the RESTORED latents (denoise.py:252-256);
//   * the scheduler is RESET per window (denoise.py:152-156), so step 0's sigma
//     is the first of a fresh schedule and not a continuation.
std::vector<std::vector<float>> Music3DenoiseChunks(
    const std::vector<float>& frame_hiddens, int64_t num_frames,
    const MiniMaxMusic3Config& config, const Music3AcousticWeights& weights,
    const Music3DenoiseOptions& options, const Music3NoiseSource& noise);

// The decode + stitch (decoders.py:75-92): each window's latents through the
// vocoder, cropped by `VocoderCropSpan`, concatenated, and CLAMPED to [-1, 1]
// (decoders.py:89 `.clamp(-1.0, 1.0)`). Returns [2, samples] channel-major;
// `*out_samples_per_channel` receives `samples`.
std::vector<float> Music3DecodeChunks(const std::vector<std::vector<float>>& latent_chunks,
                                      const MiniMaxMusic3VocoderConfig& config,
                                      const VocoderWeights& weights,
                                      int64_t* out_samples_per_channel);

// ---------------------------------------------------------------------------
// The request contract, ahead of any staging
// ---------------------------------------------------------------------------

// What one request resolves to once the family's defaults are applied. Split
// out from `Synthesize` so the contract gates without a 28.5 GB checkpoint.
struct Music3Request {
  std::string prompt;          // the assembled AR prompt (encoders.py:207-210)
  double audio_duration_s = 0.0;
  int64_t max_frames = 0;      // encoders.py:287
  int64_t num_inference_steps = 0;
  double guidance_scale = 0.0;
  int64_t seed = 0;
};

// Validate + resolve, or THROW naming the field.
//
// FOUR REFUSALS, and every one names the field rather than the family:
//   * `text` set — Music3 has no single-utterance input; `lyrics` and
//     `description` are separate because upstream normalizes them differently
//     (encoders.py:54-91). SILENTLY DROPPING IT would discard the caller's
//     words, which is the failure a shared struct makes easy;
//   * `lyrics` empty — there is nothing to sing, and upstream's
//     `_normalize_lyrics` would emit a bare "[start]" prompt;
//   * `reference_audio` supplied — Music3 has no voice cloning, and
//     `requires_reference_audio()` is FALSE for exactly that reason;
//   * `language` set — the AR prompt template has no language slot
//     (encoders.py:207-210), so honouring it is impossible and ignoring it is a
//     lie.
// Plus upstream's own two: a non-positive duration and a duration shorter than
// one frame (encoders.py:277-291), both raised by `MaxArFrames`.
Music3Request Music3ResolveRequest(const multimodal::SpeechGenParams& params,
                                   const MiniMaxMusic3ConditionEncoderConfig& config);

// ---------------------------------------------------------------------------
// The engine
// ---------------------------------------------------------------------------

// Register the family into `registry`. Additive: a registry without it behaves
// exactly as before, which is what the additivity gate asserts.
void RegisterMiniMaxMusic3SpeechFamily(multimodal::SpeechRegistry& registry);

// Register every speech family this build ships, in one place, so a caller
// (`vllm_speech_engine_load`, the server) cannot be handed a half-populated
// registry. Idempotent: calling it twice does NOT throw the registry's
// duplicate-name error.
void RegisterBuiltinSpeechFamilies(multimodal::SpeechRegistry& registry);

}  // namespace music3
}  // namespace models
}  // namespace vllm
