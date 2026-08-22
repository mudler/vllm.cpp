// See minimax_music3_speech.h for why the composition lives here and why the
// noise source is a parameter.
#include "vllm/model_executor/models/minimax_music3_speech.h"

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "vllm/model_executor/model_loader/safetensors_reader.h"
#include "vllm/model_executor/models/indextts2.h"
#include "vllm/model_executor/models/minimax_music3_llm.h"
#include "vllm/model_executor/models/music3_profile.h"
#include "vt/backend.h"
#include "vt/dtype.h"

namespace vllm {
namespace models {
namespace music3 {
namespace {

namespace fs = std::filesystem;

[[noreturn]] void Fail(const std::string& message) { throw std::runtime_error(message); }

// The pipeline class `modular_model_index.json` names for this model
// (modular_pipeline.py). Matching the CLASS rather than the directory name is
// what makes detection an inspection of the artifact.
constexpr const char* kPipelineClass = "MiniMaxMusic3ModularPipeline";

bool ReadWholeFile(const fs::path& path, std::string* out) {
  std::error_code ec;
  if (!fs::is_regular_file(path, ec) || ec) return false;
  std::ifstream in(path, std::ios::binary);
  if (!in) return false;
  std::ostringstream buffer;
  buffer << in.rdbuf();
  *out = buffer.str();
  return true;
}

}  // namespace

double Music3FrameRate(const MiniMaxMusic3ConditionEncoderConfig& config) {
  if (config.input_hop_length <= 0) {
    Fail("MiniMax-Music3: the condition encoder's `input_hop_length` must be positive");
  }
  return static_cast<double>(config.input_sampling_rate) /
         static_cast<double>(config.input_hop_length);
}

// ---------------------------------------------------------------------------
// Detection
// ---------------------------------------------------------------------------

bool Music3DetectCheckpoint(const std::string& path) {
  std::error_code ec;
  if (path.empty() || !fs::is_directory(path, ec) || ec) return false;

  std::string index;
  if (!ReadWholeFile(fs::path(path) / "modular_model_index.json", &index)) return false;
  // The class name, not the directory spelling. A repackager renames the
  // directory; the index carries what the pipeline IS.
  if (index.find(kPipelineClass) == std::string::npos) return false;

  // Every component W1 resolves must be present, so a truncated download is not
  // claimed and then refused three stages later. `vocoder` is the seventh
  // directory the index lists beside the six named components.
  for (const char* component : kMusic3Components) {
    if (!fs::is_directory(fs::path(path) / component, ec) || ec) return false;
  }
  return fs::is_directory(fs::path(path) / kMusic3VocoderComponent, ec) && !ec;
}

// ---------------------------------------------------------------------------
// The chunk plan
// ---------------------------------------------------------------------------

std::vector<Music3Chunk> Music3ChunkPlan(int64_t num_frames, const ConditionMixConfig& config) {
  if (num_frames <= 0) Fail("MiniMax-Music3: the chunk plan needs at least one AR frame");
  const std::vector<int64_t> starts = ChunkStarts(num_frames);
  std::vector<Music3Chunk> out;
  out.reserve(starts.size());
  for (const int64_t start : starts) {
    Music3Chunk chunk;
    chunk.frame_start = start;
    // denoise.py:80 — `min(start + _CHUNK_FRAMES, frame_hiddens.shape[1])`. The
    // clamp is what makes the LAST window shorter than 200 frames.
    chunk.frame_end = std::min<int64_t>(start + kChunkFrames, num_frames);
    chunk.latent_length = ConditionLatentLength(chunk.frames(), config);
    out.push_back(chunk);
  }
  return out;
}

// ---------------------------------------------------------------------------
// The noise source
// ---------------------------------------------------------------------------

Music3NoiseSource Music3SeededNoise(int64_t seed) {
  return [seed](int64_t channels, int64_t length, int64_t chunk_index) {
    if (channels <= 0 || length <= 0) {
      Fail("MiniMax-Music3: the initial latents need positive channels and length");
    }
    // The window index enters the seed so two windows of one request do not draw
    // the same noise, which upstream's single running generator also avoids.
    std::mt19937_64 engine(static_cast<uint64_t>(seed) * 0x9E3779B97F4A7C15ull +
                           static_cast<uint64_t>(chunk_index));
    std::normal_distribution<float> normal(0.0f, 1.0f);
    std::vector<float> out(static_cast<size_t>(channels * length));
    for (float& value : out) value = normal(engine);
    return out;
  };
}

// ---------------------------------------------------------------------------
// The acoustic weights
// ---------------------------------------------------------------------------

namespace {

// One acoustic tensor, refused BY NAME when its dtype is not the F32 spec §2.1
// fixes for the acoustic half rather than being silently reinterpreted.
std::vector<float> AcousticF32(const StTensor& tensor, const std::string& name) {
  if (tensor.dtype != "F32") {
    Fail("MiniMax-Music3: acoustic tensor '" + name + "' is " + tensor.dtype +
         ", and the acoustic half runs F32 (spec §2.1)");
  }
  std::vector<float> out(tensor.nbytes / sizeof(float));
  std::memcpy(out.data(), tensor.data, tensor.nbytes);
  return out;
}

// The condition encoder's FILE is fp32 while its RUNTIME is bf16 (spec §2.1's
// invariant `dtype(LM) == dtype(rvq) == dtype(cond)`), so its tensors are
// ROUNDED here rather than widened.
std::vector<float> ConditionAtRuntimeDtype(const StTensor& tensor, const std::string& name) {
  std::vector<float> out = AcousticF32(tensor, name);
  for (float& value : out) value = vt::BF16ToF32(vt::F32ToBF16(value));
  return out;
}

}  // namespace

Music3AcousticWeights Music3LoadAcousticWeights(const MiniMaxMusic3Paths& paths,
                                                const MiniMaxMusic3Config& config) {
  Music3AcousticWeights out;

  // BREAKDOWN ROWS, all SPANS: they sit inside the `load.acoustic_weights` leaf
  // and summing them would double-count it. `SafetensorsFile` is an mmap whose
  // tensors are COPIED out, so the read and the copy are interleaved and the
  // total cannot say which one it is.
  const auto cond_t0 = profile::Now();
  const SafetensorsFile condition_file = SafetensorsFile::Open(
      (fs::path(paths.condition_encoder_dir) / "diffusion_pytorch_model.safetensors").string());
  const auto condition_get = [&condition_file](const std::string& name) {
    return ConditionAtRuntimeDtype(condition_file.Get(name), name);
  };
  out.condition.layer_weight_logits = condition_get("layer_weight_logits");
  out.condition.layer_scale = condition_get("layer_scale");
  out.condition.proj_weight = condition_get("proj.weight");
  out.condition.proj_bias = condition_get("proj.bias");
  profile::AddSince("load.ac.condition", cond_t0, /*span=*/true);

  const auto voc_t0 = profile::Now();
  const SafetensorsFile vocoder_file = SafetensorsFile::Open(
      (fs::path(paths.vocoder_dir) / "diffusion_pytorch_model.safetensors").string());
  out.vocoder = VocoderWeightsFromLoader(
      config.vocoder, MiniMaxMusic3LoadVocoderWeights(config.vocoder, vocoder_file));
  profile::AddSince("load.ac.vocoder", voc_t0, /*span=*/true);

  if (paths.transformer_shards.empty()) {
    Fail("MiniMax-Music3: the transformer has no safetensors shards");
  }
  // The two halves of the fp32 DiT's 9.7 GB: reading each tensor out of the
  // mmap into a `std::map` of `std::vector<float>`, and then rebuilding the
  // weight struct from that map. They are separated because they are different
  // costs — the first touches every source page, the second is a pure host copy
  // that touches no file at all — and only the second can be blamed on the
  // loader rather than on the storage.
  std::map<std::string, std::vector<float>> dit;
  {
    profile::Timer read_timer("load.ac.dit_read", /*span=*/true);
    for (const std::string& shard : paths.transformer_shards) {
      const SafetensorsFile file = SafetensorsFile::Open(shard);
      for (const std::string& name : file.Names()) dit[name] = AcousticF32(file.Get(name), name);
    }
  }
  {
    profile::Timer build_timer("load.ac.dit_build", /*span=*/true);
    out.dit = DitWeightsFromTensors(config.transformer, dit);
  }
  return out;
}

// ---------------------------------------------------------------------------
// The DiT's device arm, selected
// ---------------------------------------------------------------------------

Music3DenoiseDeviceArm Music3SelectDitArm(vt::Queue& queue,
                                          const MiniMaxMusic3TransformerConfig& config,
                                          DitWeights& weights, bool release_host,
                                          Music3DitDeviceWeights* staged) {
  // See minimax_music3_speech.h for why this is a function and not an `if` in
  // the engine: the engine's condition is false on every runner CI owns, and
  // #1131 is what an ungateable selection costs.
  if (staged == nullptr) {
    Fail("MiniMax-Music3: the DiT arm selection needs somewhere to stage into");
  }
  Music3DenoiseDeviceArm arm;
  if (queue.device.type == vt::DeviceType::kCPU) return arm;
  // Timed HERE rather than at the call site so the span exists only when
  // something is actually staged; a CPU queue returns above and records nothing.
  profile::Timer stage_timer("acoustic.dit_staging");
  *staged = StageMusic3DitWeights(queue, config, weights, release_host);
  arm.queue = &queue;
  arm.dit = staged;
  return arm;
}

// ---------------------------------------------------------------------------
// The denoise loop
// ---------------------------------------------------------------------------

std::vector<std::vector<float>> Music3DenoiseChunks(const std::vector<float>& frame_hiddens,
                                                    int64_t num_frames,
                                                    const MiniMaxMusic3Config& config,
                                                    const Music3AcousticWeights& weights,
                                                    const Music3DenoiseOptions& options,
                                                    const Music3NoiseSource& noise,
                                                    const Music3DenoiseDeviceArm& device_arm) {
  if (!noise) Fail("MiniMax-Music3: the denoise loop needs a noise source");
  if (options.num_inference_steps <= 0) {
    Fail("MiniMax-Music3: `num_inference_steps` must be positive, got " +
         std::to_string(options.num_inference_steps));
  }
  // Half a device arm is a caller that believes it asked for the GPU and got the
  // host loops. Refused by name rather than ignored, because the failure it
  // otherwise produces is a correct song delivered thirty hours late.
  if (device_arm.half_set()) {
    Fail("MiniMax-Music3: the denoise device arm needs BOTH a queue and staged DiT weights; "
         "got only the " + std::string(device_arm.queue != nullptr ? "queue" : "weights"));
  }
  const bool on_device = device_arm.engaged();

  ConditionMixConfig mix;
  mix.condition_hidden_dim = config.condition_encoder.condition_hidden_dim;
  mix.num_condition_layers = config.condition_encoder.num_condition_layers;
  mix.out_dim = config.condition_encoder.out_dim;
  mix.input_sampling_rate = config.condition_encoder.input_sampling_rate;
  mix.input_hop_length = config.condition_encoder.input_hop_length;
  mix.output_sampling_rate = config.condition_encoder.output_sampling_rate;
  mix.output_hop_length = config.condition_encoder.output_hop_length;

  const int64_t row = mix.num_condition_layers * mix.condition_hidden_dim;
  if (static_cast<int64_t>(frame_hiddens.size()) != num_frames * row) {
    Fail("MiniMax-Music3: frame_hiddens holds " + std::to_string(frame_hiddens.size()) +
         " values, " + std::to_string(num_frames) + " frames x " + std::to_string(row) +
         " needs " + std::to_string(num_frames * row));
  }

  const int64_t channels = config.transformer.in_channels;
  const int64_t condition_dim = config.transformer.condition_dim;
  const std::vector<Music3Chunk> plan = Music3ChunkPlan(num_frames, mix);

  // The schedule is the SAME for every window (denoise.py:152-156 resets it per
  // window from the same ramp), so it is built once and walked from 0 each time.
  const FlowMatchSchedule schedule =
      FlowMatchSetTimesteps(DenoiseSigmaRamp(options.num_inference_steps), config.scheduler);

  std::vector<std::vector<float>> latent_chunks;
  latent_chunks.reserve(plan.size());
  std::vector<float> previous_latent;
  std::vector<float> previous_condition;  // [prev_len, condition_dim]
  int64_t previous_length = 0;

  for (size_t k = 0; k < plan.size(); ++k) {
    const Music3Chunk& chunk = plan[k];
    const std::vector<float> window(
        frame_hiddens.begin() + static_cast<ptrdiff_t>(chunk.frame_start * row),
        frame_hiddens.begin() + static_cast<ptrdiff_t>(chunk.frame_end * row));
    // The condition encoder runs bf16 (spec §2.1); its OUTPUT is then consumed
    // by an fp32 DiT, which is upstream's one cast at denoise.py:83.
    std::vector<float> condition;
    {
      profile::Timer condition_timer("denoise.condition_mix");
      condition = ConditionMix(window, chunk.frames(), mix, weights.condition,
                               ArCompute::kBFloat16);
    }
    const int64_t length = static_cast<int64_t>(condition.size()) / condition_dim;
    if (length * condition_dim != static_cast<int64_t>(condition.size())) {
      Fail("MiniMax-Music3: the condition mix returned a non-rectangular tensor");
    }

    // denoise.py:85-88: splice the PREVIOUS window's condition over the overlap
    // BEFORE the noise is drawn, so the noise follows the spliced condition.
    const int64_t overlap = previous_length > 0 ? std::min(previous_length, length) : 0;
    for (int64_t t = 0; t < overlap; ++t) {
      for (int64_t d = 0; d < condition_dim; ++d) {
        condition[static_cast<size_t>(t * condition_dim + d)] =
            previous_condition[static_cast<size_t>(t * condition_dim + d)];
      }
    }

    std::vector<float> latents = noise(channels, length, static_cast<int64_t>(k));
    if (static_cast<int64_t>(latents.size()) != channels * length) {
      Fail("MiniMax-Music3: the noise source returned " + std::to_string(latents.size()) +
           " values, window " + std::to_string(k) + " needs " +
           std::to_string(channels * length));
    }
    // denoise.py:122 — the blend prompt is a SNAPSHOT of the initial noise over
    // the overlap, taken before the first step overwrites those columns.
    std::vector<float> noise_prompt;
    if (overlap > 0) {
      noise_prompt.resize(static_cast<size_t>(channels * overlap));
      for (int64_t c = 0; c < channels; ++c) {
        for (int64_t t = 0; t < overlap; ++t) {
          noise_prompt[static_cast<size_t>(c * overlap + t)] =
              latents[static_cast<size_t>(c * length + t)];
        }
      }
    }

    const std::vector<float> zero_condition(condition.size(), 0.0f);
    for (int64_t step = 0; step < options.num_inference_steps; ++step) {
      const double time_value = schedule.timesteps[static_cast<size_t>(step)];
      if (overlap > 0) {
        BlendOverlap(latents, channels, length, noise_prompt, previous_latent, previous_length,
                     overlap, time_value);
      }
      // The ONLY line the device arm changes. Both CFG branches take the same
      // arm — running one on each would make the guidance mix a comparison
      // between two different numerics rather than between two conditionings.
      std::vector<float> conditional;
      std::vector<float> unconditional;
      {
        // ONE bracket over BOTH CFG branches, and the call count is therefore
        // half the DiT forwards. `calls` x 2 is the forward count the spec
        // (§14.6) quotes a per-forward figure against.
        profile::Timer dit_timer(on_device ? "denoise.dit_device" : "denoise.dit_host");
        conditional =
            on_device ? DitForwardDevice(*device_arm.queue, latents, length, condition, time_value,
                                         config.transformer, *device_arm.dit)
                      : DitForward(latents, length, condition, time_value, config.transformer,
                                   weights.dit);
        unconditional =
            on_device ? DitForwardDevice(*device_arm.queue, latents, length, zero_condition,
                                         time_value, config.transformer, *device_arm.dit)
                      : DitForward(latents, length, zero_condition, time_value,
                                   config.transformer, weights.dit);
      }
      const std::vector<float> velocity =
          ClassifierFreeGuidanceMix(conditional, unconditional, options.guidance_scale);
      latents = FlowMatchStep(latents, velocity, step, schedule);
    }

    // denoise.py:249-250: the overlap is RESTORED exactly after the loop. The
    // per-step blend approaches it but never reaches it, so skipping this leaves
    // the seam audible.
    for (int64_t c = 0; c < channels; ++c) {
      for (int64_t t = 0; t < overlap; ++t) {
        latents[static_cast<size_t>(c * length + t)] =
            previous_latent[static_cast<size_t>(c * previous_length + t)];
      }
    }

    const WindowCarrySpan carry = ChunkCarrySpan(length);
    previous_length = carry.length();
    previous_latent.assign(static_cast<size_t>(channels * previous_length), 0.0f);
    for (int64_t c = 0; c < channels; ++c) {
      for (int64_t t = 0; t < previous_length; ++t) {
        previous_latent[static_cast<size_t>(c * previous_length + t)] =
            latents[static_cast<size_t>(c * length + carry.start + t)];
      }
    }
    previous_condition.assign(static_cast<size_t>(previous_length * condition_dim), 0.0f);
    for (int64_t t = 0; t < previous_length; ++t) {
      for (int64_t d = 0; d < condition_dim; ++d) {
        previous_condition[static_cast<size_t>(t * condition_dim + d)] =
            condition[static_cast<size_t>((carry.start + t) * condition_dim + d)];
      }
    }

    latent_chunks.push_back(std::move(latents));
  }
  return latent_chunks;
}

// ---------------------------------------------------------------------------
// The decode
// ---------------------------------------------------------------------------

std::vector<float> Music3DecodeChunks(const std::vector<std::vector<float>>& latent_chunks,
                                      const MiniMaxMusic3VocoderConfig& config,
                                      const VocoderWeights& weights,
                                      int64_t* out_samples_per_channel) {
  if (latent_chunks.empty()) Fail("MiniMax-Music3: the decode needs at least one latent window");
  if (out_samples_per_channel == nullptr) {
    Fail("MiniMax-Music3: the decode needs an out_samples_per_channel pointer");
  }
  const int64_t hop = config.hop_length();
  const int64_t num_chunks = static_cast<int64_t>(latent_chunks.size());

  std::vector<std::vector<float>> left_channel;
  std::vector<std::vector<float>> right_channel;
  int64_t total = 0;
  for (int64_t k = 0; k < num_chunks; ++k) {
    const std::vector<float>& latents = latent_chunks[static_cast<size_t>(k)];
    const int64_t length = static_cast<int64_t>(latents.size()) / config.latent_channels;
    if (length * config.latent_channels != static_cast<int64_t>(latents.size())) {
      Fail("MiniMax-Music3: latent window " + std::to_string(k) + " is not rectangular");
    }
    int64_t samples = 0;
    std::vector<float> waveform;
    {
      profile::Timer vocoder_timer("vocoder.decode_window");
      waveform = VocoderDecode(latents, length, config, weights, &samples);
    }
    const WaveformCropSpan span = VocoderCropSpan(k, num_chunks, samples, hop);
    if (span.length() <= 0) {
      Fail("MiniMax-Music3: window " + std::to_string(k) + " crops to nothing (" +
           std::to_string(samples) + " samples, hop " + std::to_string(hop) + ")");
    }
    std::vector<float> l(static_cast<size_t>(span.length()));
    std::vector<float> r(static_cast<size_t>(span.length()));
    for (int64_t i = 0; i < span.length(); ++i) {
      l[static_cast<size_t>(i)] = waveform[static_cast<size_t>(span.left + i)];
      r[static_cast<size_t>(i)] = waveform[static_cast<size_t>(samples + span.left + i)];
    }
    total += span.length();
    left_channel.push_back(std::move(l));
    right_channel.push_back(std::move(r));
  }

  // decoders.py:89 — the concatenated waveform is CLAMPED to [-1, 1]. The
  // vocoder's own tanh already bounds it, so this is inert on a correct decode
  // and is mirrored anyway rather than argued away.
  std::vector<float> out(static_cast<size_t>(2 * total));
  int64_t written = 0;
  for (size_t k = 0; k < left_channel.size(); ++k) {
    for (size_t i = 0; i < left_channel[k].size(); ++i) {
      out[static_cast<size_t>(written + static_cast<int64_t>(i))] =
          std::min(1.0f, std::max(-1.0f, left_channel[k][i]));
      out[static_cast<size_t>(total + written + static_cast<int64_t>(i))] =
          std::min(1.0f, std::max(-1.0f, right_channel[k][i]));
    }
    written += static_cast<int64_t>(left_channel[k].size());
  }
  *out_samples_per_channel = total;
  return out;
}

// ---------------------------------------------------------------------------
// The request contract
// ---------------------------------------------------------------------------

Music3Request Music3ResolveRequest(const multimodal::SpeechGenParams& params,
                                   const MiniMaxMusic3ConditionEncoderConfig& config) {
  if (!params.text.empty()) {
    Fail("MiniMax-Music3: `text` is not this family's input — a music request carries "
         "`lyrics` (the sung text, with [Verse]/[Chorus] tags) and `description` (genre, "
         "BPM, key, instrumentation) as SEPARATE fields, because upstream normalizes them "
         "differently (encoders.py:54-91). Move the text into one of the two rather than "
         "having it silently dropped");
  }
  if (params.lyrics.empty()) {
    Fail("MiniMax-Music3: `lyrics` is required — there is nothing to sing, and an empty "
         "lyric normalizes to a bare '[start]' prompt");
  }
  if (!params.reference_audio.empty()) {
    Fail("MiniMax-Music3: `reference_audio` is not supported — this family has no voice "
         "cloning and no reference conditioning, which is why requires_reference_audio() "
         "is false. Supply `description` instead");
  }
  if (!params.language.empty()) {
    Fail("MiniMax-Music3: `language` is not supported — the prompt template has no "
         "language slot (encoders.py:207-210), so it can be neither honoured nor honestly "
         "ignored. State the language in `description` or in the lyrics themselves");
  }

  // ZERO means "omitted, take the family default"; NEGATIVE is an explicit,
  // impossible value and is REFUSED. Collapsing the two turns upstream's
  // `audio_duration must be positive` (encoders.py:277-278) into a silent 60 s
  // song — which is exactly what this branch did until the gate caught it.
  if (params.audio_duration_s < 0.0) {
    Fail("MiniMax-Music3: `audio_duration_s` must be positive, got " +
         std::to_string(params.audio_duration_s) +
         " (leave it 0 to take the family's 60 s default)");
  }
  if (params.num_inference_steps < 0) {
    Fail("MiniMax-Music3: `num_inference_steps` must be positive, got " +
         std::to_string(params.num_inference_steps) +
         " (leave it 0 to take the family's 30-step default)");
  }

  Music3Request out;
  out.audio_duration_s =
      params.audio_duration_s > 0.0 ? params.audio_duration_s : kMusic3DefaultDurationSeconds;
  // MaxArFrames raises upstream's own two errors (encoders.py:277-291).
  out.max_frames = MaxArFrames(out.audio_duration_s, Music3FrameRate(config));
  out.num_inference_steps = params.num_inference_steps > 0 ? params.num_inference_steps
                                                           : kMusic3DefaultInferenceSteps;
  // NEGATIVE means default, because 0 is a legal guidance scale.
  out.guidance_scale = params.guidance_scale >= 0.0 ? params.guidance_scale : kDitGuidanceScale;
  out.seed = params.seed;
  // Assembled LAST: it raises on an empty description, and the field refusals
  // above are the ones that name a `SpeechGenParams` field.
  out.prompt = AssembleArPrompt(params.description, params.lyrics);
  if (static_cast<int64_t>(out.prompt.size()) > kMaxPromptTokens) {
    // A byte count is an UPPER bound on the token count for any BPE, so a prompt
    // that passes this can still be too long — the tokenizer-side check belongs
    // with the AR head. This one catches the case that is already impossible.
    Fail("MiniMax-Music3: the assembled prompt is " + std::to_string(out.prompt.size()) +
         " bytes, past the checkpoint's " + std::to_string(kMaxPromptTokens) +
         "-token ceiling however it tokenizes");
  }
  return out;
}

// ---------------------------------------------------------------------------
// The engine
// ---------------------------------------------------------------------------

namespace {

class Music3SpeechEngine final : public multimodal::SpeechEngine {
 public:
  Music3SpeechEngine(std::string path, int32_t device) : path_(std::move(path)) {
    // FIRST, before a single file is opened. The device refusal is free and the
    // checkpoint resolution is not, and a caller that asked for a device this
    // build cannot serve should not learn it after three minutes of NAS I/O.
    // Same ordering rule as `Synthesize`'s "validate FIRST".
    const vt::DeviceType type =
        multimodal::SpeechEngineDeviceType(device, kMusic3SpeechFamily);

    paths_ = MiniMaxMusic3ResolveCheckpoint(path_);
    config_ = MiniMaxMusic3LoadConfig(paths_);
    // The dtype invariant of spec §2.1, enforced BEFORE anything stages, so a
    // violating configuration is named here rather than surfacing as a torch-
    // shaped type error inside a forward.
    MiniMaxMusic3CheckRuntimeDtypes(
        MiniMaxMusic3ResolveRuntimeDtypes(MiniMaxMusic3DtypePolicy::kBf16ArFp32Acoustic));

    if (type == vt::DeviceType::kCPU) {
      // The DEFAULT arm, and the one every Music3 gate was taken on. Constructed
      // exactly as the hardcoded queue this replaced.
      queue_ = vt::Queue{vt::Device{vt::DeviceType::kCPU, 0}, nullptr};
    } else {
      // The device context is created BEFORE any weight is read — the
      // unified-memory load recipe minimax_h3_video.cpp:349-353 records, and
      // Jetson Thor (sm_110, ~122 GB UNIFIED) is exactly the box it was written
      // for: the driver's reservation has to land before host allocations fill
      // the one pool both sides draw from.
      queue_ = vt::GetBackend(type).CreateQueue();
    }
  }

  std::string family() const override { return kMusic3SpeechFamily; }
  // WHERE this engine resolved to run, reported rather than echoed back from
  // the request: "device 1 was asked for" and "device 1 was granted" are
  // different facts, and a benchmark that conflates them measures the CPU arm
  // twice.
  vt::Device device() const override { return queue_.device; }
  int64_t sample_rate() const override { return config_.vocoder.sampling_rate; }
  // FALSE, and it is load-bearing rather than a default: Music3 conditions on
  // text alone (spec §4.1), where IndexTTS-2 has no text-only synthesis at all.
  // A server consults this to refuse before staging, so getting it backwards
  // would reject every valid music request.
  bool requires_reference_audio() const override { return false; }

  multimodal::SpeechResult Synthesize(const multimodal::SpeechGenParams& params) override {
    // Validate FIRST: every field refusal is free, and a caller learns its
    // request was malformed without waiting for 28.5 GB to stage.
    profile::Begin();
    profile::Mark("synthesize.enter");
    const Music3Request request = Music3ResolveRequest(params, config_.condition_encoder);
    profile::Count("request.max_frames", request.max_frames);
    profile::Count("request.steps", request.num_inference_steps);

    // ── the AUTOREGRESSIVE half (encoders.py) ────────────────────────────────
    //
    // SCOPED, so the 8.6B language model and the 0.65B depth decoder (~18.5 GB
    // together) are released before the fp32 acoustic half (~10 GB) stages.
    // Upstream does the same by hand rather than by luck — encoders.py:302-309
    // drives the offload hooks itself, with the comment that both models must be
    // co-resident for the loop and neither may evict the other. Only
    // `frame_hiddens` crosses the boundary.
    std::vector<float> frame_hiddens;
    int64_t frames = 0;
    int64_t calls = 0;
    {
      // The AR stage's language model runs through `vt`, so the queue is the
      // only thing that decides where. CPU is what W2 shipped and what every
      // gate for this row was taken on; a device arm is a queue, not a fork —
      // and that is now literally true rather than an intention, because the
      // queue this line used to construct as a constant is the engine's
      // `queue_` and the ONLY thing the device selector changes.
      //
      // WHAT MOVES: the 8.6B `Qwen3ForCausalLM` half, through the shared
      // `Qwen3DenseModel::ForwardEmbeds` five registrations already ride, AND —
      // since #1309 — the 0.65B RVQ depth decoder, which was 48.4 % of a run
      // (spec §19.1). WHAT DOES NOT: the depth decoder's projection, audio
      // heads and feedback embedding, ~1.6 % of that stage and owed by §19.7;
      // and the whole acoustic half's host reference loops — see
      // minimax_music3_ar.h and vocoder1d.h for which pieces are owed and why.
      //
      // NON-const, because the depth arm below STAGES OUT OF IT.
      const auto load_t0 = profile::Now();
      Music3ArWeights ar = Music3LoadArWeights(paths_, config_);
      profile::AddSince("load.ar_weights", load_t0);
      profile::Mark("ar.weights_loaded");
      const std::vector<int32_t> prompt_ids = ar.Encode(request.prompt);

      // THE PRODUCTION SELECTION for the depth decoder, on the SAME switch the
      // DiT arm rides: `--speech-device 1` resolves `queue_` to the platform's
      // device, and a non-CPU queue takes the device arm. There is no separate
      // flag and no environment variable, because a capability behind an option
      // nothing turns on is the shape `.agents/reachability.md` calls dead.
      //
      // The rule itself lives in `Music3SelectDepthArm` rather than in an `if`
      // here, and that placement is the #1131 repair: the condition `queue_` has
      // to satisfy is false on every runner CI owns, so a branch written at this
      // line is unreachable from any gate, while the function is driven by
      // `test_minimax_music3_ar` on both sides of it. The DiT block below still
      // carries the untestable shape and #1131 still owns it.
      //
      // `release_host` is TRUE: the staged tensors are the ONLY thing the host
      // append loop reads, and it is not called when the arm is engaged. The
      // projection, the audio embeddings and the audio heads — which this stage
      // still reads on the host — are not staged and are not released.
      Music3DepthDeviceWeights staged_depth;
      const Music3DepthDeviceArm depth_arm = Music3SelectDepthArm(
          queue_, ar.depth_config, ar.depth, /*release_host=*/true, &staged_depth);
      Music3ArResult generated;
      {
        profile::Timer ar_timer("ar.TOTAL_loop", /*span=*/true);
        generated = Music3GenerateFrameHiddens(prompt_ids, request.max_frames, ar,
                                               Music3SeededSampler(request.seed), queue_,
                                               depth_arm);
      }
      profile::Mark("ar.loop_done");
      frame_hiddens = std::move(generated.frame_hiddens);
      frames = generated.frames;
      calls = generated.calls;
    }
    (void)calls;

    // ── the ACOUSTIC half (before_denoise.py / denoise.py / decoders.py) ─────
    //
    // NON-const, because the device arm below STAGES OUT OF IT. The DiT is
    // 9.7 GB of fp32 and Jetson Thor's ~122 GB is UNIFIED — host and device draw
    // on one pool — so uploading while the host copy is still held is a real
    // 19.4 GB peak on the only box this arm runs on, and that box reboots
    // instead of OOM-killing (.agents/environment.md). `StageMusic3DitWeights`
    // drops each host tensor as it lands, so the peak is one tensor over the
    // 9.7 GB, not twice it.
    profile::Mark("ar.weights_released");
    profile::Count("ar.frames", frames);
    const auto acoustic_t0 = profile::Now();
    Music3AcousticWeights acoustic = Music3LoadAcousticWeights(paths_, config_);
    profile::AddSince("load.acoustic_weights", acoustic_t0);
    profile::Mark("acoustic.weights_loaded");
    Music3DenoiseOptions options;
    options.num_inference_steps = request.num_inference_steps;
    options.guidance_scale = request.guidance_scale;

    // Staged ONCE per request, outside every loop in `Music3DenoiseChunks`. A
    // 45 s clip runs the DiT 660 times over 11 windows; a per-window upload
    // would move 9.7 GB eleven times and a per-step one 660 times, either of
    // which costs more than the compute it enables.
    //
    // Not staged in the CONSTRUCTOR, unlike the queue: the acoustic weights are
    // deliberately loaded per request and released with the request, so that the
    // 18.5 GB autoregressive half and the 10 GB acoustic half are never
    // co-resident (upstream drives the same split by hand, encoders.py:302-309).
    // Staging follows the weights, not the engine.
    //
    // THE PRODUCTION SELECTION, on the SAME switch the depth arm rides:
    // `--speech-device 1` resolves `queue_` to the platform's device, and a
    // non-CPU queue takes the device arm. There is no separate flag and no
    // environment variable, because a capability behind an option nothing turns
    // on is the shape `.agents/reachability.md` calls dead.
    //
    // The rule itself lives in `Music3SelectDitArm` rather than in an `if` here,
    // and that placement is the #1131 repair this row landed: the condition
    // `queue_` has to satisfy is false on every runner CI owns, so a branch
    // written at this line is unreachable from any gate, while the function is
    // driven by `test_minimax_music3_acoustic` on both sides of it. The CALL
    // below is what remains ungated, and `music3-dit-arm-reachability.md`
    // `## Owed` says so rather than leaving it to be found again.
    //
    // `release_host` is TRUE: once the DiT is resident, the staged tensors are
    // the only thing the denoise loop reads.
    Music3DitDeviceWeights staged_dit;
    const Music3DenoiseDeviceArm arm = Music3SelectDitArm(
        queue_, config_.transformer, acoustic.dit, /*release_host=*/true, &staged_dit);
    profile::Mark("acoustic.dit_staged");
    std::vector<std::vector<float>> chunks;
    {
      profile::Timer denoise_timer("denoise.TOTAL", /*span=*/true);
      chunks = Music3DenoiseChunks(frame_hiddens, frames, config_, acoustic, options,
                                   Music3SeededNoise(request.seed), arm);
    }
    profile::Mark("denoise.done");
    profile::Count("denoise.windows", static_cast<int64_t>(chunks.size()));

    int64_t samples = 0;
    multimodal::SpeechResult out;
    {
      profile::Timer decode_timer("vocoder.TOTAL", /*span=*/true);
      out.samples = Music3DecodeChunks(chunks, config_.vocoder, acoustic.vocoder, &samples);
    }
    profile::Mark("vocoder.done");
    // spec §1.1: the vocoder's NATIVE rate, resample-free. The 32 kHz form is a
    // downstream delivery transform and the caller's decision, which is exactly
    // what `SpeechResult`'s own contract says `sample_rate` means.
    out.sample_rate = config_.vocoder.sampling_rate;
    out.channels = kMusic3Channels;
    if (samples <= 0 ||
        static_cast<int64_t>(out.samples.size()) != kMusic3Channels * samples) {
      Fail("MiniMax-Music3: the decode returned " + std::to_string(out.samples.size()) +
           " values for " + std::to_string(samples) + " frames of " +
           std::to_string(kMusic3Channels) + "-channel audio");
    }
    profile::Count("output.samples_per_channel", samples);
    profile::Report("MiniMax-Music3 synthesize");
    return out;
  }

 private:
  std::string path_;
  MiniMaxMusic3Paths paths_;
  MiniMaxMusic3Config config_;
  // Built ONCE, in the constructor, and reused by every request: on an
  // accelerator this owns the stream, and creating one per `Synthesize` would
  // put a context creation inside the request path.
  vt::Queue queue_{};
};

}  // namespace

void RegisterMiniMaxMusic3SpeechFamily(multimodal::SpeechRegistry& registry) {
  multimodal::SpeechFamilyRegistration reg;
  reg.name = kMusic3SpeechFamily;
  reg.detect = [](const multimodal::SpeechModelParams& params) {
    return Music3DetectCheckpoint(params.path);
  };
  reg.load = [](const multimodal::SpeechModelParams& params)
      -> std::unique_ptr<multimodal::SpeechEngine> {
    return std::make_unique<Music3SpeechEngine>(params.path, params.device);
  };
  registry.Register(std::move(reg));
}

void RegisterBuiltinSpeechFamilies(multimodal::SpeechRegistry& registry) {
  const std::vector<std::string> present = registry.families();
  const auto has = [&present](const char* name) {
    return std::find(present.begin(), present.end(), name) != present.end();
  };
  // Idempotent by construction: `SpeechRegistry::Register` THROWS on a duplicate
  // name (deliberately — two claimants sharing one listed name defeats the
  // never-guess guarantee), and a process-global registry can be populated from
  // more than one entry point.
  if (!has("indextts2")) RegisterIndexTts2SpeechFamily(registry);
  if (!has(kMusic3SpeechFamily)) RegisterMiniMaxMusic3SpeechFamily(registry);
}

}  // namespace music3
}  // namespace models
}  // namespace vllm
