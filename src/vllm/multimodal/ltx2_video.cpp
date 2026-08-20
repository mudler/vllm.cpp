// LTX-2.5 behind the generalized video seam — implementation. See
// include/vllm/multimodal/ltx2_video.h for the port map and the three refusals.
//
// Row: MODEL-DIFFUSION-LTX25, .agents/specs/ltx-2-5.md phase L7. Issue #435.
#include "vllm/multimodal/ltx2_video.h"

// W0 of LTX25-DEVICE-RESIDENCY (#1010): the phase instrument. Every scope
// below is a production call site — the table is written on the shipped
// default, not behind a flag, because the runs whose profile the campaign
// needs are the long ones nobody knew to instrument in advance.
#include "vllm/multimodal/render_phase_log.h"

#include <algorithm>
#include <atomic>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

#include "vllm/model_executor/model_loader/safetensors_reader.h"
#include "vllm/model_executor/models/device_pool.h"  // ActivePool(b)/DevicePool::Drain
#include "vllm/model_executor/models/ltx2.h"
#include "vllm/model_executor/models/ltx2_audio_input.h"
#include "vllm/model_executor/models/ltx2_audio_vae.h"
#include "vllm/model_executor/models/ltx2_conditioning.h"
#include "vllm/model_executor/models/ltx2_connector.h"
#include "vllm/model_executor/models/ltx2_denoisers.h"
#include "vllm/model_executor/models/ltx2_device.h"
#include "vllm/model_executor/models/ltx2_dfr.h"
#include "vllm/model_executor/models/ltx2_image_preprocess.h"
#include "vllm/model_executor/models/ltx2_loader.h"
#include "vllm/model_executor/models/ltx2_pipeline.h"
#include "vllm/model_executor/models/ltx2_samplers.h"
#include "vllm/model_executor/models/ltx2_retake.h"
#include "vllm/model_executor/models/ltx2_t2a.h"
#include "vllm/model_executor/models/ltx2_text_encoder.h"
#include "vllm/model_executor/models/ltx2_upsampler.h"
#include "vllm/model_executor/models/ltx2_tiling.h"
#include "vllm/model_executor/models/ltx2_video_vae.h"
#include "vllm/model_executor/models/ltx2_video_vae_encoder.h"
#include "vllm/model_executor/models/minimax_h3.h"
#include "vllm/platforms/interface.h"  // CurrentPlatform() — which accelerator, if any
#include "vllm/tokenizer/tokenizer.h"

namespace vllm::multimodal {
namespace {

[[noreturn]] void Fail(const std::string& message) {
  throw std::runtime_error("ltx-2.5 video: " + message);
}

std::string ReadFileBytes(const std::string& field, const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) Fail(field + ": cannot open " + path);
  return std::string((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
}

void WriteFileBytes(const std::string& path, const std::string& bytes) {
  std::ofstream out(path, std::ios::binary);
  if (!out) Fail("cannot write " + path);
  out.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
  if (!out) Fail("short write " + path);
}

// A JSON object read from a FILE, for the one configuration a checkpoint may not
// carry itself (`dit_config_path`). Both failure modes are refusals by name: a
// path that will not open, and bytes that are not a JSON object. Returning an
// empty object on either would be indistinguishable from an empty config, which
// is exactly the silent default this extra exists to remove.
nlohmann::json ReadJsonFile(const std::string& field, const std::string& path) {
  const std::string bytes = ReadFileBytes(field, path);
  nlohmann::json parsed;
  try {
    parsed = nlohmann::json::parse(bytes);
  } catch (const std::exception& e) {
    Fail(field + ": '" + path + "' is not JSON (" + e.what() + ")");
  }
  if (!parsed.is_object()) Fail(field + ": '" + path + "' is not a JSON object");
  return parsed;
}

std::vector<float> ReadF32File(const std::string& field, const std::string& path) {
  const std::string bytes = ReadFileBytes(field, path);
  if (bytes.size() % sizeof(float) != 0) {
    Fail(field + ": '" + path + "' is " + std::to_string(bytes.size()) +
         " bytes, not a whole number of little-endian f32 values");
  }
  std::vector<float> out(bytes.size() / sizeof(float));
  std::memcpy(out.data(), bytes.data(), bytes.size());
  return out;
}

// ── the noise stream ────────────────────────────────────────────────────────
//
// Upstream draws every tensor from ONE seeded `torch.Generator`
// (distilled.py:214-215) and the ancestral loop draws from a second one offset
// by `ANCESTRAL_NOISE_SEED_OFFSET` (:70-73, :177-183). Reproducing torch's
// Mersenne/Philox stream bit-exactly would decide WHICH sample comes out, not
// whether the pipeline is right — the same call MiniMax-H3 made and recorded
// (minimax_h3.h:1895-1897). So this is a documented splitmix64 + Box-Muller
// stream: deterministic, seedable, drawn in the SAME ORDER upstream draws
// (video before audio, one draw per state per step), and NOT torch's.
//
// What that costs is stated rather than hidden: a clip rendered here is a
// different sample from the same distribution as upstream's, so it is not
// comparable to an upstream render frame by frame. Sample-level comparison
// against the binding oracle needs the noise supplied from outside, which is
// exactly why every brick below L7 takes its noise as an argument.
class SplitMixGaussian {
 public:
  explicit SplitMixGaussian(uint64_t seed) : state_(seed) {}

  void Fill(float* out, int64_t count) {
    for (int64_t i = 0; i < count; ++i) out[i] = Next();
  }
  std::vector<float> Draw(int64_t count) {
    std::vector<float> out(static_cast<size_t>(count));
    Fill(out.data(), count);
    return out;
  }

 private:
  uint64_t NextBits() {
    state_ += 0x9E3779B97F4A7C15ULL;
    uint64_t z = state_;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
  }
  double Uniform() {
    // (0, 1): Box-Muller takes a log of the first draw, so 0 must be excluded.
    return (static_cast<double>(NextBits() >> 11) + 0.5) * (1.0 / 9007199254740992.0);
  }
  float Next() {
    if (have_spare_) {
      have_spare_ = false;
      return spare_;
    }
    const double u1 = Uniform();
    const double u2 = Uniform();
    const double r = std::sqrt(-2.0 * std::log(u1));
    const double theta = 6.283185307179586476925286766559 * u2;
    spare_ = static_cast<float>(r * std::sin(theta));
    have_spare_ = true;
    return static_cast<float>(r * std::cos(theta));
  }

  uint64_t state_ = 0;
  bool have_spare_ = false;
  float spare_ = 0.0F;
};

// The VAE decoder's `torch.randn` source (ltx2_video_vae.h:152-161), consumed in
// CALL ORDER — which is what makes the decode reproducible on both sides.
class EngineNoiseStream : public Ltx2NoiseStream {
 public:
  explicit EngineNoiseStream(uint64_t seed) : gen_(seed) {}
  std::vector<float> Draw(int64_t count) override { return gen_.Draw(count); }

 private:
  SplitMixGaussian gen_;
};

// ── one modality's state through the loop (ltx-core types.py LatentState) ───
// Every buffer is PATCHIFIED: [tokens, width]. `mask` and `clean` are what
// `post_process_latent` (utils/helpers.py:462-464) blends between.
struct StreamState {
  int64_t tokens = 0;
  int64_t width = 0;                 // channels per token
  std::vector<float> latent, clean;  // [tokens, width]
  std::vector<float> mask;           // [tokens], patchified denoise mask
  std::vector<double> positions;     // [n_pos_dims, tokens, 2]
  // `_first_frame_keyframes_mask` (tools.py:186-196), [tokens]. VIDEO only, and
  // populated on EVERY generation — the rule has no branch on whether a keyframe
  // was supplied. Empty on the audio stream, whose args preprocessor upstream
  // builds with no keyframes_embedding_provider (model.py:333).
  std::vector<float> keyframes_mask;
};

// ── StreamState <-> Ltx2LatentState (row LTX25-TOKEN-APPEND, issue #930) ────
//
// The conditioning items take `Ltx2LatentState`; the loop runs on `StreamState`.
// ONE statement of the mapping, in both directions, because the first-frame arm
// used to open-code the copy and an appending arm needs three more fields than
// it carried — `positions`, `keyframes_mask` and a `tokens` that comes BACK
// changed. A second open-coded copy is how one of them gets forgotten, and a
// forgotten `keyframes_mask` is invisible to every shape check downstream.
//
// THE POSITIONS ROUND TRIP IS EXACT and that is not luck. `StreamState` holds
// them as `double` only because the DiT's `positions` field takes one; every
// value in it was produced by widening a `float` (see where the temporal axis is
// divided by fps, below), and upstream's own positions are `float32`
// (tools.py:169-174). So double -> float -> double reproduces the bits.
Ltx2LatentState ToLatentState(const StreamState& s, int64_t pos_dims) {
  Ltx2LatentState out;
  out.tokens = s.tokens;
  out.width = s.width;
  out.pos_dims = pos_dims;
  out.latent = s.latent;
  out.clean = s.clean;
  out.mask = s.mask;
  // Explicit cast: StreamState stores positions as double for the DiT surface,
  // Ltx2LatentState keeps float32. Range-assign would narrow implicitly and
  // trip MSVC C4244 under /WX (mudler/vllm.cpp#968).
  out.positions.resize(s.positions.size());
  for (size_t i = 0; i < s.positions.size(); ++i) {
    out.positions[i] = static_cast<float>(s.positions[i]);
  }
  out.keyframes_mask = s.keyframes_mask;
  return out;
}

void FromLatentState(const Ltx2LatentState& in, StreamState* s) {
  s->tokens = in.tokens;
  s->width = in.width;
  s->latent = in.latent;
  s->clean = in.clean;
  s->mask = in.mask;
  s->positions.resize(in.positions.size());
  for (size_t i = 0; i < in.positions.size(); ++i) {
    s->positions[i] = static_cast<double>(in.positions[i]);
  }
  s->keyframes_mask = in.keyframes_mask;
}

// `post_process_latent` (utils/helpers.py:462-464):
//   denoised * mask + clean * (1 - mask)
// The mask is PER TOKEN and the latent is per token x channel, so the mask
// broadcasts along the channel axis exactly as torch's trailing-axis rule does.
// TEMPLATED ON THE VALUE TYPE because upstream calls this at two widths and the
// res_2s loop reaches both: at the model dtype on a denoiser result
// (samplers.py:305, :390, :441) and at `hp` on a sample inside
// `_inject_sde_noise` (samplers.py:203). One implementation, two
// instantiations; a second copy of the blend is the shape this campaign has
// recorded going wrong. The mask is 0 or 1 on every LTX-2.5 path, so the result
// is exactly one operand or the other and the two widths agree.
template <typename Value>
std::vector<Value> PostProcessLatent(const std::vector<Value>& denoised,
                                     const StreamState& state) {
  std::vector<Value> out(denoised.size());
  for (int64_t t = 0; t < state.tokens; ++t) {
    const Value m = static_cast<Value>(state.mask[static_cast<size_t>(t)]);
    for (int64_t c = 0; c < state.width; ++c) {
      const size_t i = static_cast<size_t>(t * state.width + c);
      out[i] = denoised[i] * m + static_cast<Value>(state.clean[i]) * (static_cast<Value>(1) - m);
    }
  }
  return out;
}

// `GaussianNoiser.__call__` (components/noisers.py:30-37), on the PATCHIFIED
// state. The second lerp is the one a port gets backwards; `Ltx2GaussianNoise`
// already implements both and is gated, so this only broadcasts the per-token
// mask onto the per-token x channel latent before calling it.
void ApplyGaussianNoise(StreamState& state, const std::vector<float>& noise, float noise_scale) {
  std::vector<float> broadcast_mask(state.latent.size());
  for (int64_t t = 0; t < state.tokens; ++t) {
    for (int64_t c = 0; c < state.width; ++c) {
      broadcast_mask[static_cast<size_t>(t * state.width + c)] = state.mask[static_cast<size_t>(t)];
    }
  }
  state.latent = Ltx2GaussianNoise(state.latent.data(), state.clean.data(), broadcast_mask.data(),
                                   noise.data(), static_cast<int64_t>(state.latent.size()),
                                   noise_scale);
}

// `timesteps_from_mask` (utils/helpers.py:494-503): the per-token timestep is
// `denoise_mask * sigma`, so a conditioned token is at timestep 0 while an
// unconditioned one is at the schedule's sigma.
std::vector<float> TimestepsFromMask(const StreamState& state, float sigma) {
  std::vector<float> out(static_cast<size_t>(state.tokens));
  for (int64_t t = 0; t < state.tokens; ++t) out[static_cast<size_t>(t)] = state.mask[static_cast<size_t>(t)] * sigma;
  return out;
}

// `X0Model.forward` (model/transformer/model.py:601-604) composed with
// `to_denoised` (utils.py:38-50): the DiT emits a VELOCITY and the loop consumes
// a denoised prediction, `x - sigma * v`, with the PER-TOKEN sigma. Getting the
// per-token part wrong (using the scalar schedule sigma for every token) is
// invisible until a request carries conditioned tokens, and then it re-noises
// them.
std::vector<float> ToDenoised(const std::vector<float>& sample, const std::vector<float>& velocity,
                              const std::vector<float>& timesteps, int64_t tokens, int64_t width) {
  std::vector<float> out(sample.size());
  for (int64_t t = 0; t < tokens; ++t) {
    const double sigma = timesteps[static_cast<size_t>(t)];
    for (int64_t c = 0; c < width; ++c) {
      const size_t i = static_cast<size_t>(t * width + c);
      out[i] = static_cast<float>(static_cast<double>(sample[i]) -
                                  sigma * static_cast<double>(velocity[i]));
    }
  }
  return out;
}

// W0 of LTX25-DEVICE-RESIDENCY (#1010): the phase table lands BESIDE the frames
// it explains, on the shipped default.
//
// A file rather than a console line, because #1040 is made of exactly the
// evidence that existed only on a host that stopped answering, and a render that
// took two hours is not one somebody was watching. An IO failure here is
// reported and swallowed: a render must not fail because its instrument could
// not write, and a silent success would be worse than either.
void WritePhaseLog(const std::string& output_dir, const std::string& family,
                   const std::string& device, std::string* out_path);

std::string JoinPath(const std::string& dir, const std::string& leaf) {
  if (dir.empty()) return leaf;
  return dir.back() == '/' ? dir + leaf : dir + "/" + leaf;
}

void WritePhaseLog(const std::string& output_dir, const std::string& family,
                   const std::string& device, std::string* out_path) {
  const std::string path = JoinPath(output_dir, "phase-log.json");
  std::string why;
  if (phase::PhaseLog::Instance().WriteJson(path, family, device, &why)) {
    *out_path = path;
    return;
  }
  std::fprintf(stderr, "[ltx2] the phase table was not written: %s\n", why.c_str());
}

int64_t ExtraInt(const std::map<std::string, std::string>& extras, const std::string& key,
                 int64_t fallback) {
  const std::string raw = VideoExtra(extras, key);
  if (raw.empty()) return fallback;
  try {
    size_t consumed = 0;
    const long long value = std::stoll(raw, &consumed);
    if (consumed != raw.size()) throw std::invalid_argument("trailing");
    return static_cast<int64_t>(value);
  } catch (const std::exception&) {
    Fail("the extra '" + key + "' is '" + raw + "', which is not an integer");
  }
}

// The same, for a SECONDS-valued knob. Separate from `ExtraInt` rather than
// folded into it, because the two disagree about what "3" means to a parser and
// silently truncating `audio_start_time=1.5` to 1 would window the wrong second
// of a take and still render. `consumed != size` catches the trailing-garbage
// case that `stod` otherwise accepts.
double ExtraDouble(const std::map<std::string, std::string>& extras, const std::string& key,
                   double fallback) {
  const std::string raw = VideoExtra(extras, key);
  if (raw.empty()) return fallback;
  try {
    size_t consumed = 0;
    const double value = std::stod(raw, &consumed);
    if (consumed != raw.size()) throw std::invalid_argument("trailing");
    if (!std::isfinite(value)) throw std::invalid_argument("not finite");
    return value;
  } catch (const std::exception&) {
    Fail("the extra '" + key + "' is '" + raw + "', which is not a finite number of seconds");
  }
}

// The IC-LoRA strength (utils/args.py:600-611). Upstream's `LoraAction` parses
// it as a plain float and applies no range clamp, so neither does this: a
// negative or >1 strength is a legitimate, if unusual, request that upstream
// honours, and refusing it here would diverge. What IS refused is a value that
// is not a number at all, which upstream's `float()` would raise on too.
//
// Not `ExtraDouble` above, and deliberately: that one reports "not a finite
// number of SECONDS", which is the wrong noun for a strength, and it defaults a
// missing key while this one is only ever called on a key that is present.
double ParseLoraStrength(const std::string& raw) {
  try {
    size_t consumed = 0;
    const double value = std::stod(raw, &consumed);
    if (consumed != raw.size()) throw std::invalid_argument("trailing");
    if (!std::isfinite(value)) throw std::invalid_argument("non-finite");
    return value;
  } catch (const std::exception&) {
    Fail("the extra '" + std::string(kLtx2LoraStrengthExtra) + "' is '" + raw +
         "', which is not a finite number");
  }
}

// The one key this family DEFINES and does not SERVE. `Ltx2DurationPredict` is
// ported and gated as a brick (`ltx2_duration_head.h`), but nothing here
// constructs one, so a supplied path names a file the engine never opens.
constexpr char kLtx2DurationHeadPathExtra[] = "duration_head_path";

// Every extra key this family DEFINES. An extra outside this set is refused
// rather than ignored, for the same reason H3 refuses one
// (minimax_h3_video.cpp): a mistyped knob that is silently dropped renders the
// DEFAULT and looks like the feature not working.
//
// DEFINED IS NOT THE SAME AS SERVED, and conflating the two was #611: nine of
// these ten reach a reader, and `duration_head_path` reached none, so supplying a
// duration head substituted the recipe default in silence — the failure mode this
// very list exists to prevent, one level in. It stays in the list because the
// family DOES define the key and DOES know what it means; `CheckUnservedExtras`
// refuses it by name instead, which is a different and truer message than
// "unknown load extra". The full audit is in
// .agents/specs/ltx25-retire-dead-arms.md §2.1.
//
// The first hand-written set of these anchors named nine lines that were readers
// of NOTHING, in this very file, and a later merge moved the real ones again. So
// they are no longer trusted: the list below is derived from this file on every
// run and compared, and the failure prints the replacement to paste in.
// READER ANCHORS (derived and gated by test_ltx2_video):
// 889 899 900 968 1064 1080 1128 1221 1247 1355 1397 1439 1441

const char* const kKnownLoadExtras[] = {
    kLtx2AudioPromptEmbedsExtra, kLtx2PipelineKindExtra,   kLtx2ModelVersionExtra,
    kLtx2AllowUnportedExtra,     kLtx2MaxPhaseExtra,       kLtx2DitConfigPathExtra,
    kLtx2PromptValidRowsExtra,   kLtx2EncoderConfigPathExtra,
    "upsampler_path",            kLtx2DurationHeadPathExtra,
    kLtx2LoraPathExtra,          kLtx2LoraStrengthExtra,
    kLtx2NegativePromptEmbedsExtra, kLtx2NegativeAudioPromptEmbedsExtra,
};

// FNV-1a over the raw bytes of a float buffer — the `Ltx2ConditioningTrace`
// digest. It is deliberately a function of the BYTES rather than of a reduction
// like a sum or a mean: a sum cannot see a permutation and a mean cannot see two
// compensating changes, and the question this instrument answers is exactly
// "are these the same numbers in the same places".
uint64_t DigestF32(const std::vector<float>& values) {
  uint64_t h = 1469598103934665603ULL;
  const auto* bytes = reinterpret_cast<const unsigned char*>(values.data());
  const size_t n = values.size() * sizeof(float);
  for (size_t i = 0; i < n; ++i) {
    h ^= bytes[i];
    h *= 1099511628211ULL;
  }
  return h;
}

double AbsMax(const std::vector<float>& values) {
  double m = 0.0;
  for (const float v : values) m = std::max(m, std::abs(static_cast<double>(v)));
  return m;
}

void CheckKnownExtras(const std::map<std::string, std::string>& extras) {
  for (const auto& kv : extras) {
    bool known = false;
    for (const char* name : kKnownLoadExtras) {
      if (kv.first == name) known = true;
    }
    if (!known) {
      std::string listing;
      for (const char* name : kKnownLoadExtras) listing += std::string(listing.empty() ? "" : ", ") + name;
      Fail("unknown load extra '" + kv.first + "'. This family defines: " + listing);
    }
  }
}

// A key this family DEFINES but does not SERVE, refused BY NAME when supplied
// (#611). The alternative — accepting it — is the worst of the three options:
// worse than refusing, and worse than not defining the key, because the caller
// pointed at a specific file and got the recipe default with no diagnostic.
//
// Deliberately NOT the "unknown load extra" path above. That message says the
// family does not define the key, which is false here and would send the reader
// looking for a typo instead of for the unported head.
void CheckUnservedExtras(const std::map<std::string, std::string>& extras) {
  const std::string duration_head = VideoExtra(extras, kLtx2DurationHeadPathExtra);
  if (!duration_head.empty()) {
    Fail("the '" + std::string(kLtx2DurationHeadPathExtra) + "' extra names '" + duration_head +
         "', but the duration head is NOT WIRED into this engine: `Ltx2DurationPredict` is ported "
         "and gated as a brick (ltx2_duration_head.h, upstream duration_head.py:89-118) and "
         "nothing here constructs one, so that file would never be opened and an AUTO duration "
         "would fall back to the recipe default. Give 'num_frames', or 'duration' (exact "
         "arithmetic against the recipe frame rate), instead. Refused rather than ignored; "
         "recorded as owed in .agents/specs/ltx25-retire-dead-arms.md (#611).");
  }
}

// `detect_model_version` normalizes the separator before parsing
// (utils/constants.py:161), and the recipe table is keyed on the two-component
// spelling ("2.5"), not on the checkpoint's three-component "2.5.0". Reduce it
// with the SAME parser the table's own callers use, so "2.5.0", "2.5" and
// "2.5-rc1" all resolve to one row instead of two of them missing it.
std::string RecipeVersionKey(const std::string& declared) {
  const std::vector<int64_t> parsed = Ltx2ParseModelVersion(declared);
  if (parsed.empty()) {
    Fail("the DiT checkpoint declares model_version '" + declared +
         "', whose numeric prefix is empty (loader/helpers.py:62-81). A recipe is resolved "
         "from an EXACT (kind, version) pair and never defaulted.");
  }
  std::string key = std::to_string(parsed[0]);
  if (parsed.size() > 1) key += "." + std::to_string(parsed[1]);
  return key;
}

// `EmbeddingsProcessor`'s two connector calls plus its output-mask contract —
// ONE statement of it, because two callers now reach it and they must not answer
// "what does the connector do here" differently: the LOAD-time prompt-embeds
// path, and the PER-REQUEST path that phase L13 added on top of the text tower.
// Before L13 there was one caller and the logic sat inline; a second copy is
// exactly the shape this project has recorded as producing two rules.
//
// THE WEIGHTS LIVE AND DIE INSIDE THIS CALL, and that is the whole reason it
// takes a `SafetensorsFile` rather than materialized weights. 129 tensors per
// stream is ~1.61G video + ~0.40G audio parameters at the shipped widths —
// about 8 GB of f32 together — and holding them for an engine's lifetime on a
// 119 GB unified-memory box that reboots rather than OOM-killing is 8 GB spent
// on a module that runs once per request over 1024 rows. A diffusion request is
// minutes; re-reading the DiT file is not the cost that matters here.
//
// THE f32 IS AN ANNOTATED ESCAPE, not an inherited default. Upstream runs this
// module at the model dtype, so f32 here is WIDER — the polarity AGENTS.md says
// a value gate cannot catch. It is taken because `Ltx2ConnectorForward` is L5's
// declared PARITY dtype and this is the arm its goldens cover, and its output is
// narrowed to the stream dtype on the first upload like every other activation.
Ltx2ConnectorEmbeddings RunConnector(const SafetensorsFile& dit_file,
                                     const Ltx2ConnectorConfig& video_cfg,
                                     const Ltx2ConnectorConfig& audio_cfg,
                                     const std::vector<float>& video_in,
                                     const std::vector<float>& audio_in,
                                     const std::vector<float>& additive, int64_t rows) {
  const Ltx2VaeWeights video_weights = Ltx2LoadConnectorWeights(dit_file, video_cfg);
  const Ltx2VaeWeights audio_weights = Ltx2LoadConnectorWeights(dit_file, audio_cfg);
  Ltx2ConnectorEmbeddings encoded = Ltx2ConnectorCreateEmbeddings(
      video_cfg, video_weights, video_in.data(), audio_cfg, audio_weights, audio_in.data(),
      additive.data(), /*batch=*/1, rows);
  // The processor returns the mask the DiT's cross-attention is supposed to
  // honour (embeddings_processor.py:89). `Ltx2ModalityInput` carries no context
  // mask, so a mask with a masked position would be silently dropped — the DiT
  // would attend over register-free padding as if it were caption.
  //
  // THIS LOOP IS UNREACHABLE ON EVERY INPUT EITHER REFERENCE PRODUCES, and
  // saying which case reaches it is how the claim stays checkable. It is NOT the
  // `num_learnable_registers = 0` case, which an earlier version of this comment
  // named and which cannot get here: with registers disabled
  // `Ltx2ConnectorForward` passes the caller's ADDITIVE mask straight through,
  // and `_to_binary_mask`'s `encoded_mask < 0.000001`
  // (embeddings_processor.py:46-48) is satisfied by BOTH values an additive mask
  // holds — 0.0 and -finfo(f32).max — so the binary mask is one everywhere there
  // too. With registers enabled :152 returns `torch.zeros_like(mask)` and the
  // answer is one everywhere for the same reason. What would reach it is a
  // connector whose output mask carries a value at or above +1e-6, which no path
  // in `ltx_core` or `diffusers` emits today. It is kept as a guard on that
  // future, refused by name rather than ignored, and it is deliberately not
  // gated: a test would have to fabricate a mask neither reference can produce.
  for (const float m : encoded.mask) {
    if (m == 1.0f) continue;
    Fail(
        "the embeddings connector returned a cross-attention mask with masked "
        "positions, and `Ltx2ModalityInput` carries no context mask to pass it through. "
        "No path in either reference emits such a mask — `_to_binary_mask` is one "
        "everywhere for both values an additive mask holds — so this is a connector "
        "whose output mask this port does not model. Refusing rather than dropping "
        "the mask, which would condition the DiT on unmasked padding.");
  }
  return encoded;
}

}  // namespace

// ---------------------------------------------------------------------------
// Detection
// ---------------------------------------------------------------------------

// Does this checkpoint set hold an LTX-2.5 DiT? The discriminator is the DUAL
// PATCHIFY PROJECTION, `patchify_proj.weight` + `audio_patchify_proj.weight` —
// the two names phase L2's contract binds first (`EnumerateLtx2DitTensors`) and
// the two every shipped arm carries, ComfyUI prefix or not.
//
// It CANNOT collide with MiniMax-H3's, which keys on `video_patch_proj.weight` +
// `audio_patch_proj.weight` (minimax_h3_video.cpp:820-831). The names are
// different words, not a prefix of one another, so neither detector can see the
// other's checkpoint however either file was repackaged.
//
// Deliberately NOT keyed on the file extension, on the directory layout, or on
// the presence of a `model.diffusion_model.` prefix: which container a checkpoint
// was repackaged into, and whether the repackager kept ComfyUI's prefix, say
// nothing about which model it holds. Both shipped LTX-2.5 DiTs carry the
// prefix; a de-prefixed re-export must still be found.
bool DetectLtx2Video(const VideoModelParams& params) {
  std::vector<std::string> names;
  std::string why;
  if (!ReadVideoCheckpointTensorNames(params.dit_path, &names, &why)) return false;
  const std::string prefix = kLtx2DitCheckpointPrefix;
  bool video_patchify = false, audio_patchify = false;
  for (const std::string& n : names) {
    std::string bare = n;
    if (bare.size() > prefix.size() && bare.compare(0, prefix.size(), prefix) == 0) {
      bare = bare.substr(prefix.size());
    }
    if (bare == "patchify_proj.weight") video_patchify = true;
    if (bare == "audio_patchify_proj.weight") audio_patchify = true;
    if (video_patchify && audio_patchify) return true;
  }
  return false;
}

// ---------------------------------------------------------------------------
// The engine
// ---------------------------------------------------------------------------

struct Ltx2VideoEngine::Impl {
  VideoModelParams params;
  // CPU, or the accelerator index `params.device - 1` names on whichever device
  // type the platform seam resolves. Phase L8 made the second real: the DiT is
  // staged with `Ltx2StreamDitToDevice` and driven by `Ltx2DitForwardDevice`, so
  // a non-zero handle now denotes a device-resident forward.
  vt::Device device;
  // The stream dtype the DiT was staged at, and the one the forward computes in.
  // bf16 on an accelerator — upstream resolves ONE model dtype and every layer
  // inherits it, and bf16 is what `Ltx2StreamDitToDevice` puts on the device.
  // f32 on the CPU, which is the L2 parity forward's declared dtype.
  vt::DType compute_dtype = vt::DType::kF32;
  bool on_device = false;
  std::optional<vt::Queue> queue;

  Ltx2DitCheckpoint dit;
  // The adapter set the LOAD supplied, kept so the phase loop can put the DiT
  // into the state each phase asks for (`Ltx2PhaseRecipe::loras`). Upstream
  // instead builds a second `DiffusionStage` per adapter set
  // (a2vid_two_stage.py:103 and :115); this engine holds one DiT and
  // re-materializes the adapter's target tensors at the boundary, because a
  // second resident weight set is 18.7-39 GB and one GB10 has 119 GB with no
  // swap. `Ltx2RebindDitLoras` carries the whole argument.
  //
  // The SPECS only — the adapter file itself is re-read per rebind rather than
  // held, since its A/B factors are its whole payload and keeping them resident
  // would spend most of what the second-weight-set shape was rejected for.
  Ltx2DitLoadOptions dit_options;
  std::string model_version, pipeline_kind;
  Ltx2PipelineRecipe recipe;
  int64_t max_phase = -1;  // -1 => every phase

  Ltx2VideoDecoderKind video_kind = Ltx2VideoDecoderKind::kConv;
  Ltx2ConvVideoDecoderConfig video_cfg;
  Ltx2VaeWeights video_weights;

  // The ENCODER half of the same file (row LTX25-IMAGE-COND, issue #644). Its
  // absence is what every conditioning arm was refused for: before this row the
  // load below materialized `Ltx2VideoVaeDecoderKeyRules()` alone and no encoder
  // key filter existed anywhere in the tree, so `Ltx2ConvVideoEncode` — ported
  // and gated since phase L11 — had no weights to run on.
  //
  // A Comfy-split `vae/` file may carry the decoder alone, so this is OPTIONAL
  // and its absence is reported by name at the request rather than guessed at.
  bool has_video_encoder = false;
  Ltx2ConvVideoEncoderConfig video_encoder_cfg;
  Ltx2VaeWeights video_encoder_weights;

  Ltx2AudioDecoderConfig audio_cfg;
  Ltx2VaeWeights audio_weights;
  Ltx2VocoderBweConfig vocoder_cfg;
  Ltx2VaeWeights vocoder_weights;

  // The ANALYSIS half, for audio-to-video (row LTX25-A2V-AUDIO-INPUT, #922).
  // Present only when the audio VAE checkpoint carries `audio_vae.encoder.`
  // tensors; a decoder-only checkpoint leaves this false and `audio_path` is
  // then refused BY NAME rather than rendering an unconditioned clip.
  bool has_audio_encoder = false;
  Ltx2AudioEncoderLoad audio_encoder_cfg;
  Ltx2VaeWeights audio_encoder_weights;

  bool has_upsampler = false;
  Ltx2UpsamplerConfig upsampler_cfg;
  Ltx2VaeWeights upsampler_weights;

  // Conditioning. `has_encoder` is true exactly when `encoder_path` named a
  // Gemma-4 text tower and it materialized; see the header's item 2.
  bool has_encoder = false;
  // What the DiT's cross-attention consumes. When the checkpoint carries the
  // embeddings connector these are the CONNECTOR's output; without one they are
  // the supplied prompt embeds verbatim, which is what every render before L9c
  // did on every checkpoint.
  std::vector<float> video_prompt_embeds, audio_prompt_embeds;
  int64_t prompt_tokens = 0;
  // The NEGATIVE half of the same fallback (row LTX25-GUIDED-VIDEO, #1092).
  // Empty when the load supplied none, which is what makes a guider that asks
  // for the unconditional forward a refusal rather than a silent reuse of the
  // positive context.
  std::vector<float> negative_video_prompt_embeds, negative_audio_prompt_embeds;

  // The connector's CONFIGURATION is kept; its WEIGHTS are not. They are ~8 GB
  // of f32 at the shipped widths (ltx2_loader.h), the conditioning they process
  // is resolved once at load, and this box reboots rather than OOM-killing — so
  // they are loaded, used and dropped inside one scope below.
  bool has_connector = false;
  Ltx2ConnectorConfig video_connector_cfg, audio_connector_cfg;
  int64_t prompt_valid_rows = 0;

  // ── the text tower (phase L13) ────────────────────────────────────────────
  //
  // RESIDENT, unlike the connector weights, and the asymmetry is deliberate.
  // A prompt arrives PER REQUEST, so the tower has to survive the load: at the
  // shipped 12B that is ~24 GB of host bf16 and the header says so. The
  // connector is ~8 GB of f32 and is re-read from the DiT file inside one scope
  // per request instead, so the steady state is the tower alone. Both numbers
  // matter on a 119 GB unified-memory box that reboots rather than OOM-killing.
  //
  // `Ltx2GemmaTower` and `tok::Tokenizer` are held by value / by pointer rather
  // than rebuilt: rebuilding the tower per request would dequantize 337 NVFP4
  // modules on every generation.
  std::unique_ptr<Ltx2GemmaTower> tower;
  std::unique_ptr<tok::Tokenizer> tokenizer;
  Ltx2GemmaSpecialIds gemma_ids;
  Ltx2TextEncoderWeights caption_projections;
  Ltx2TextFeatureConfig feature_cfg;

  Ltx2ConditioningTrace trace;

  std::mutex mutex;
};

Ltx2VideoEngine::Ltx2VideoEngine() = default;
Ltx2VideoEngine::Ltx2VideoEngine(Ltx2VideoEngine&&) noexcept = default;
Ltx2VideoEngine& Ltx2VideoEngine::operator=(Ltx2VideoEngine&&) noexcept = default;
Ltx2VideoEngine::~Ltx2VideoEngine() = default;

std::string Ltx2VideoEngine::family() const { return kLtx2VideoFamily; }
vt::Device Ltx2VideoEngine::device() const { return impl_->device; }
bool Ltx2VideoEngine::has_encoder() const { return impl_->has_encoder; }
bool Ltx2VideoEngine::has_prompt_embeds() const { return !impl_->video_prompt_embeds.empty(); }
const std::string& Ltx2VideoEngine::model_version() const { return impl_->model_version; }
const std::string& Ltx2VideoEngine::pipeline_kind() const { return impl_->pipeline_kind; }
const Ltx2DitParams& Ltx2VideoEngine::dit_params() const { return impl_->dit.params; }
Ltx2ConditioningTrace Ltx2VideoEngine::last_conditioning() const {
  std::lock_guard<std::mutex> guard(impl_->mutex);
  return impl_->trace;
}

std::unique_ptr<Ltx2VideoEngine> Ltx2VideoEngine::Load(const VideoModelParams& params) {
  if (params.dit_path.empty()) Fail("dit_path is required");
  CheckKnownExtras(params.extras);
  CheckUnservedExtras(params.extras);

  auto engine = std::unique_ptr<Ltx2VideoEngine>(new Ltx2VideoEngine());
  engine->impl_ = std::make_unique<Impl>();
  Impl& im = *engine->impl_;
  im.params = params;

  // ── W0: the timeline starts HERE, not at the first generation (#1010) ─────
  //
  // The load is render 0 because it is a phase of the render in every sense the
  // campaign cares about: the spike measures ~7.5 minutes of DiT staging paid at
  // the front of every render and every gate run, and a table that started at
  // `Generate` would report that time as somebody else's.
  phase::PhaseLog::Instance().Begin();
  phase::PhaseLog::Instance().SetRender(0);
  const phase::Scope load_span("load", /*span=*/true);

  // ── where this engine runs (phase L8) ─────────────────────────────────────
  //
  // `device` is 0 for the CPU and 1 + <accelerator index> for an accelerator,
  // which is the mapping the seam already documents. Phase L7 REFUSED anything
  // but 0, because the f32-only forward and the bf16-only staging did not meet;
  // phase L8 is the forward that closes that, so the refusal is gone and the
  // handle now means what it says.
  //
  // WHICH accelerator is the PLATFORM's question, not this model file's. This
  // asked `TryGetBackend(kCUDA)` — the same defect work row M3a repaired in
  // `src/vllm/entrypoints/model_loader.cpp::SelectQueueForModel` (the full path
  // matters, there is also a src/vllm/model_executor/model_loader/ DIRECTORY and
  // the bare file name sends a reader there), where a hardcoded
  // `GetBackend(kCUDA)` was the one line standing between a complete non-NVIDIA
  // backend and running a model. `CurrentPlatform()` walks the probe order
  // {kCUDA, kROCM, kXPU, kVULKAN, kMETAL, kTENSTORRENT, kCPU}
  // (src/vllm/platforms/platform.cpp:62-64) and returns the first one REGISTERED
  // (:91-98) — and a platform registers only where its own probe found a device,
  // e.g. src/vllm/platforms/cuda.cpp:136-138 returns early on a box with the CUDA
  // toolkit and no usable GPU — so on a CUDA box this resolves EXACTLY the device
  // the hardcoded lookup did.
  // Nothing below this line names a device either: `Ltx2StreamDitToDevice` and
  // `Ltx2DitForwardDevice` drive `vt::Queue` and the op table, so a backend that
  // registers those ops reaches this forward with no edit here.
  //
  // What is NOT gone is the refusal to fake it: if this build registers no
  // accelerator backend, the load is refused BY NAME rather than served the CPU
  // forward behind an accelerator-looking handle. That substitution is exactly
  // what would make every later timing and every "it ran on the GPU" claim false.
  im.on_device = params.device != 0;
  if (im.on_device) {
    const vllm::platforms::Platform& platform = vllm::platforms::CurrentPlatform();
    const vt::DeviceType accelerator = platform.device_type();
    if (accelerator == vt::DeviceType::kCPU ||
        vt::TryGetBackend(accelerator) == nullptr) {
      Fail("device " + std::to_string(params.device) +
           " asks for an accelerator, but no accelerator backend is registered in this "
           "build (the platform seam resolves to '" +
           std::string(vt::DeviceTypeName(accelerator)) +
           "'). The LTX-2.5 device-resident forward is present (Ltx2DitForwardDevice); "
           "what is missing is the backend. Refusing rather than running the CPU forward "
           "behind an accelerator handle.");
    }
    // The THIRD question, which the seam's own precedent asks and this file did
    // not (#659). "Is there an accelerator" and "is a backend registered" are
    // both true on a PARTIAL backend — Metal registers 15 of 75 ops, Tenstorrent
    // a comparable slice — and both name exactly two text architectures in their
    // `supports_model_architecture` allow-lists (src/vllm/platforms/metal.cpp:70,
    // src/vllm/platforms/tenstorrent.cpp:55). Before the seam landed, such a
    // build asked `TryGetBackend(kCUDA)`, got nullptr, and REFUSED BY NAME; after
    // it, it is handed a queue and dies later inside a kernel bind with a shape
    // error that says nothing about what is missing. CUDA and CPU are unaffected:
    // `supports_model_architecture` defaults to true (interface.h:263) and is a
    // claim only a partial backend ever narrows.
    //
    // The refusal above argues that serving the CPU forward behind an accelerator
    // handle "would make every later timing and every 'it ran on the GPU' claim
    // false". A partial backend that binds and dies is the same thing one level
    // down: a device claim this build cannot honour.
    //
    // The key is the FAMILY string — this lane's stable registry name
    // (`VideoModelParams::family`) — because the diffusion engines are reached
    // through `LoadVideoEngine`, not through ModelRegistry's HF `architectures`.
    if (!platform.supports_model_architecture(kLtx2VideoFamily)) {
      Fail("device " + std::to_string(params.device) + " resolves to platform '" +
           std::string(vt::DeviceTypeName(accelerator)) +
           "', and that platform DECLINES the architecture '" +
           std::string(kLtx2VideoFamily) +
           "' (Platform::supports_model_architecture): it is a PARTIAL backend that has "
           "not registered the kernels this model needs. The build is partial, not "
           "broken. Refusing by name rather than binding a queue that would die inside a "
           "kernel bind with an error that names none of this.");
    }
    // `vt::CreateQueue(Device)`, NOT `Backend::CreateQueue()`. backend.h:212-217
    // records the method as a "temporary index-0 migration shim" and says new
    // adapter code must use the free function, and the difference is not
    // cosmetic: the shim creates the queue on device 0 and relabelling its
    // `device.index` afterwards MOVES THE LABEL, not the stream. On a multi-GPU
    // host `device = 2` would then run every kernel on GPU 0 while every
    // residency check in this file reported index 1 and agreed with itself. GB10
    // has one GPU so it cannot be reached there, which is exactly why it has to
    // be right before a second device exists.
    const int32_t index = static_cast<int32_t>(params.device - 1);
    im.device = vt::Device{accelerator, index};
    if (vt::TryGetBackend(im.device) == nullptr) {
      Fail("device " + std::to_string(params.device) + " names " +
           std::string(vt::DeviceTypeName(accelerator)) + " device index " +
           std::to_string(index) +
           ", and no backend is registered for it. Refusing rather than creating a queue "
           "on device 0 and labelling it with an index nothing runs on.");
    }
    im.queue = vt::CreateQueue(im.device);
    // W0: what the phase table's device column MEANS on this arm. The driver's
    // own in-use figure, read through the backend seam rather than by naming a
    // vendor here — nothing below the device resolution above names one.
    //
    // ON CUDA IT ANSWERS -1 TODAY, and that is #1126, not a defect in this line.
    // `CudaBackend` does not override `DeviceMemoryInfo`; ROCm is the only
    // backend that does (`src/vt/rocm/rocm_backend.hip:358`). Wiring CUDA in
    // would wake `Gemma4MoE`'s device-expert LRU, which is dead on CUDA for
    // exactly that reason (`include/vllm/platforms/interface.h:68-72`), so it is
    // a behaviour change with its own measurement and it is not an instrument's
    // to make. The column reports -1 rather than 0, because a byte count of zero
    // and a byte count nobody took are different facts.
    //
    // Installed with the queue rather than at the first phase, so every phase
    // after the device is resolved carries the column.
    {
      const vt::DeviceType probe_type = im.device.type;
      phase::PhaseLog::Instance().SetDeviceProbe([probe_type]() -> int64_t {
        vt::Backend* backend = vt::TryGetBackend(probe_type);
        if (backend == nullptr) return -1;
        size_t free_bytes = 0;
        size_t total_bytes = 0;
        if (!backend->DeviceMemoryInfo(&free_bytes, &total_bytes)) return -1;
        if (total_bytes < free_bytes) return -1;
        return static_cast<int64_t>(total_bytes - free_bytes);
      });
    }
    // bf16: upstream resolves ONE model dtype and every layer inherits it, and it
    // is what `Ltx2StreamDitToDevice` stages. f32 on an accelerator would move
    // twice the bytes to reach a gate dtype, which is the polarity this project
    // inverted deliberately.
    im.compute_dtype = vt::DType::kBF16;
  } else {
    im.device = vt::Device{};
    im.compute_dtype = vt::DType::kF32;
  }

  // ── the DiT ───────────────────────────────────────────────────────────────
  const SafetensorsFile dit_file = SafetensorsFile::Open(params.dit_path);
  // ON THE IMPL, not a local: the phase loop re-reads these to put the DiT into
  // the adapter state each phase declares (`Ltx2PhaseRecipe::loras`).
  Ltx2DitLoadOptions& dit_options = im.dit_options;
  dit_options.allow_unported_modules = VideoExtra(params.extras, kLtx2AllowUnportedExtra) == "1";
  // On the CPU, f32 is what `Ltx2DitForward` requires: it is the PARITY dtype,
  // not a widening of a bf16 path. On an accelerator nothing is widened at all —
  // `Ltx2StreamDitToDevice` dequantizes and uploads ONE TENSOR AT A TIME, so peak
  // residency is the device copy plus one tensor rather than two whole models.
  dit_options.widen_to_f32 = !im.on_device;
  // The IC-LoRA adapter, fused into the weights as they are materialized. This
  // is the production call site for the whole `ltx2_lora.h` family: deleting it
  // makes the adapter unreachable, which is what the reachability mutation in
  // the row's spec §5.3 checks.
  const std::string lora_path = VideoExtra(params.extras, kLtx2LoraPathExtra);
  const std::string lora_strength = VideoExtra(params.extras, kLtx2LoraStrengthExtra);
  if (lora_path.empty() && !lora_strength.empty()) {
    Fail("'" + std::string(kLtx2LoraStrengthExtra) + "' was given without '" +
         std::string(kLtx2LoraPathExtra) +
         "'. A strength with no adapter fuses nothing, and silently doing nothing is what "
         "this refusal exists to prevent.");
  }
  if (!lora_path.empty()) {
    Ltx2LoraSpec spec;
    spec.path = lora_path;
    if (!lora_strength.empty()) spec.strength = ParseLoraStrength(lora_strength);
    dit_options.loras.push_back(std::move(spec));
  }
  {
    // W0: the phase the campaign's W2 and W3 both act on. It covers the
    // materialization AND the per-tensor device staging, because from the
    // outside they are one wait.
    const phase::Scope dit_phase("load.dit");
    im.dit = im.on_device ? Ltx2StreamDitToDevice(*im.queue, dit_file, dit_options)
                          : Ltx2LoadDitFromSafetensors(dit_file, dit_options);
  }

  // ── the config the SHAPES cannot see ──────────────────────────────────────
  //
  // FOUND AT L7 AND REPAIRED HERE. `Ltx2LoadDitFromSafetensors` derives its
  // geometry with `ParseLtx2DitParamsFromManifest`, which reads SHAPES — the way
  // a ComfyUI checkpoint carrying no config has to be read. A config states
  // things no shape encodes:
  //
  //     "frequencies_precision": "float64"          -> double_precision_rope
  //     "av_ca_timestep_scale_multiplier": 1000.0   -> default is 1
  //     "positional_embedding_max_pos": [20,2048,2048], "..._theta": 10000.0
  //     "norm_eps", "timestep_scale_multiplier", "use_middle_indices_grid"
  //
  // Each moves every RoPE angle or every modulation while leaving the tensor
  // set byte-identical, so the manifest path resolves a DIFFERENT MODEL from the
  // same file and nothing downstream can tell. `ParseLtx2DitParams` already
  // mirrors `LTXModelConfigurator.from_metadata` (model_configurator.py:19-83)
  // for exactly this; it was simply never reached.
  //
  // AND ONLY ONE OF THE TWO SHIPPED LTX-2.5 DiTs CARRIES ONE. An earlier revision
  // of this comment said "both shipped LTX-2.5 DiTs DO carry one". That was FALSE
  // and it mattered, because the copy it was false about is the one phases L1-L6
  // gated against and the one L8 ran on the GPU. Read from the files on the NAS,
  // 2026-08-12:
  //
  //   Lightricks/LTX-2.5 ...-transformer-nvfp4.safetensors
  //       __metadata__ = ['config','gemma_source_checkpoint','license',
  //                       'model_version']                       -> declares one
  //   vonkaiser/LTX-2.5-FP8-NVFP4 ltx-2.5-22b-distilled-fp8.safetensors
  //       has __metadata__ key: FALSE                            -> declares NONE
  //
  // So the adoption below never executed for the FP8 arm, and its DiT silently
  // took `av_ca_timestep_scale_multiplier = 1` (ltx2.h:106) and
  // `double_precision_rope = false` (ltx2.h:112) against LTX-2.5's declared 1000
  // and float64. That is a different model, not a different precision, and
  // nothing said so.
  //
  // THE REPAIR IS THAT A DiT WITHOUT A CONFIG IS REFUSED, not defaulted. The
  // caller supplies one through `dit_config_path` and it is adopted through the
  // IDENTICAL weight-contract check the declared path uses. `model_version` was
  // already treated this way one block down — never defaulted, supplied by an
  // extra when the file carries none — and this is the same rule applied to the
  // configuration that decides the arithmetic rather than the schedule.
  //
  // A config is adopted only when it produces the IDENTICAL weight contract —
  // which is what proves it describes THIS file rather than another checkpoint's
  // config pasted into it. A disagreement is refused by name.
  const std::string config_path = VideoExtra(params.extras, kLtx2DitConfigPathExtra);
  const bool declares_config = dit_file.Metadata().count("config") != 0;
  if (declares_config && !config_path.empty()) {
    Fail("the DiT checkpoint declares its own __metadata__[\"config\"] and the '" +
         std::string(kLtx2DitConfigPathExtra) + "' extra names '" + config_path +
         "'. Refusing rather than preferring one: they decide values no shape can see "
         "(frequencies_precision, av_ca_timestep_scale_multiplier, the "
         "positional-embedding bounds), so the wrong choice renders with the wrong RoPE "
         "instead of failing. Drop the extra to use the checkpoint's own config.");
  }
  if (!declares_config && config_path.empty()) {
    Fail("the DiT checkpoint declares no __metadata__[\"config\"][\"transformer\"], and no '" +
         std::string(kLtx2DitConfigPathExtra) +
         "' extra was supplied. The tensor SHAPES resolve the geometry but not the values "
         "no shape encodes: double_precision_rope would default to false and "
         "av_ca_timestep_scale_multiplier to 1, where LTX-2.5 declares float64 and 1000. "
         "Both move every RoPE angle and every audio<->video modulation, so defaulting them "
         "renders a DIFFERENT MODEL confidently. This is the shape the shipped vonkaiser FP8 "
         "DiT is in: it carries no __metadata__ at all. Supply the config rather than "
         "inheriting a default that contradicts the model family.");
  }
  // Hoisted out of the block it used to live in because the EMBEDDINGS CONNECTOR
  // is configured from the SAME object: its geometry, its RoPE bounds and its
  // gating are `connector_*` keys of the DiT's transformer config, and reading
  // them from a second source would let the two disagree.
  const nlohmann::json dit_config =
      declares_config ? Ltx2ReadCheckpointConfig(dit_file)
                      : ReadJsonFile(kLtx2DitConfigPathExtra, config_path);
  {
    const std::string source =
        declares_config
            ? std::string("the DiT checkpoint's own __metadata__[\"config\"][\"transformer\"]")
            : "the '" + std::string(kLtx2DitConfigPathExtra) + "' file '" + config_path + "'";
    // `Ltx2AdoptDeclaredDitParams` (ltx2_loader.h) is the ONE place the adoption
    // rule lives, because the device gate drives `Ltx2StreamDitToDevice` without
    // this engine and owes the same check; two copies would be two rules.
    const Ltx2DitParams declared =
        Ltx2AdoptDeclaredDitParams(dit_config, im.dit.params, source);
    im.dit.params = declared;
  }

  // ── the embeddings connector (phase L9c) ──────────────────────────────────
  //
  // WHAT THIS CLOSES. Until L9c the conditioning this engine handed the DiT's
  // cross-attention was the prompt-embeds file VERBATIM. Upstream never does
  // that: `EmbeddingsProcessor` runs an 8-layer 1-D transformer over the caption
  // projections before the DiT sees them (embeddings_processor.py:70-95), and
  // that module SHIPS IN THE DiT FILE — 129 tensors per stream, which this
  // loader used to refuse as "unported" and then step over. The bricks existed
  // (`Ltx2ConnectorForward`, landed at L5, gated against upstream on five arms);
  // nothing called them outside a test. This is the call.
  //
  // PRESENCE DECIDES, not a config flag. The connector is applied exactly when
  // the checkpoint carries it, which is the same rule `DetectLtx2Video` uses on
  // the DiT itself: which modules a file HOLDS is a fact, and a flag saying
  // otherwise would be a second, disagreeable authority. A checkpoint carrying
  // one stream's connector and not the other is refused, because rendering the
  // video stream through eight transformer layers and the audio stream through
  // none conditions the two halves of one clip on two different things.
  const bool has_video_connector =
      Ltx2CheckpointHasConnector(dit_file, Ltx2ConnectorStream::kVideo);
  const bool has_audio_connector =
      Ltx2CheckpointHasConnector(dit_file, Ltx2ConnectorStream::kAudio);
  if (has_video_connector != has_audio_connector) {
    Fail(std::string("the DiT checkpoint carries the ") +
         (has_video_connector ? "video" : "audio") +
         " embeddings connector but not the other stream's. Both conditioning paths run "
         "through one `EmbeddingsProcessor` upstream (embeddings_processor.py:60-68 refuses "
         "an audio connector without audio features and vice versa), so half a connector "
         "would condition one modality on eight transformer layers and the other on none.");
  }
  im.has_connector = has_video_connector;
  if (im.has_connector) {
    im.video_connector_cfg = Ltx2ParseConnectorConfig(dit_config, Ltx2ConnectorStream::kVideo);
    im.audio_connector_cfg = Ltx2ParseConnectorConfig(dit_config, Ltx2ConnectorStream::kAudio);
    // The connector is dimension-PRESERVING: it consumes the caption projection's
    // output and hands the DiT's cross-attention the same width. Asserted rather
    // than assumed — a connector whose inner_dim disagrees with the stream it
    // feeds would still forward, and the mismatch would surface as a wrong-shaped
    // GEMM deep inside the DiT rather than as a load refusal.
    if (im.video_connector_cfg.inner_dim() != im.dit.params.cross_attention_dim) {
      Fail("the video embeddings connector is " +
           std::to_string(im.video_connector_cfg.inner_dim()) +
           " wide but the DiT's cross-attention takes " +
           std::to_string(im.dit.params.cross_attention_dim));
    }
    if (im.audio_connector_cfg.inner_dim() != im.dit.params.audio_cross_attention_dim) {
      Fail("the audio embeddings connector is " +
           std::to_string(im.audio_connector_cfg.inner_dim()) +
           " wide but the DiT's audio cross-attention takes " +
           std::to_string(im.dit.params.audio_cross_attention_dim));
    }
  }

  // ── the recipe ────────────────────────────────────────────────────────────
  const std::string declared_version = Ltx2ReadCheckpointModelVersion(dit_file);
  const std::string override_version = VideoExtra(params.extras, kLtx2ModelVersionExtra);
  if (!declared_version.empty() && !override_version.empty() &&
      RecipeVersionKey(declared_version) != RecipeVersionKey(override_version)) {
    Fail("the DiT checkpoint declares model_version '" + declared_version + "' but the '" +
         kLtx2ModelVersionExtra + "' extra says '" + override_version +
         "'. Refusing rather than preferring one: they resolve DIFFERENT sigma schedules, "
         "and the wrong one renders a video instead of failing.");
  }
  const std::string version = !declared_version.empty() ? declared_version : override_version;
  if (version.empty()) {
    Fail("the DiT checkpoint declares no __metadata__[\"model_version\"] and no '" +
         std::string(kLtx2ModelVersionExtra) +
         "' extra was supplied, so no recipe can be resolved. A recipe is never defaulted "
         "(ltx2_recipes.py:170-175).");
  }
  im.model_version = RecipeVersionKey(version);
  im.pipeline_kind = VideoExtra(params.extras, kLtx2PipelineKindExtra, "distilled_two_stage");
  im.recipe = ResolveLtx2PipelineRecipe(im.pipeline_kind, im.model_version);
  // ── the adapter a two-stage pipeline cannot run without (#1117) ───────────
  //
  // `--distilled-lora` is `required=True` on the parser `A2VidPipelineTwoStage`
  // selects (utils/args.py:1140-1155, reached through `default_2_stage_arg_parser`
  // at `:1123` from a2vid_two_stage.py:311), and the reason is what stage 2 is:
  // a THREE-sigma refinement (`:164`) that only the distilled weights can
  // complete. Run it on a checkpoint carrying no adapter and it returns a clip
  // of the right size, the right frame count and the right sample rate.
  //
  // Keyed on `recipe.requires_distilled_lora` rather than on the kind STRING, so
  // the next recipe off this parser inherits it — `ti2vid_two_stages` (#1093)
  // and `keyframe_interpolation` (#1096) are both already waiting.
  //
  // WHAT THIS FLAG SAYS is only that the load must CARRY an adapter. WHICH
  // PHASE RUNS IT is `Ltx2PhaseRecipe::loras`, and it is no longer missing:
  // this block used to close with "this engine fuses at load into ONE weight
  // set, so the adapter reaches both phases. Owed by #1118". #1118 is CLOSED,
  // by `4ae0f54ab` (row LTX25-PHASE-LORA), which added that field and
  // `Ltx2RebindDitLoras` and gave every recipe off this parser a stage 1 on the
  // base weights — `loras=tuple(loras)` (a2vid_two_stage.py:107,
  // ti2vid_two_stages.py:140) against `(*loras, *distilled_lora)` (`:114`,
  // `:151`). Rewritten rather than deleted, because a reader who finds the old
  // wording in git history needs to know it came true. #1151.
  //
  // THE ANCHOR BELOW IS THE PARSER, NOT ONE PIPELINE'S STAGE 2, and that is the
  // second half of #1151. This refusal is keyed on the flag precisely so the
  // next recipe inherits it, so a message hard-coding `a2vid_two_stage.py`'s
  // line numbers would name the caller's own pipeline in one sentence and cite
  // a different pipeline's source in the next. `utils/args.py:1140-1155` is
  // `default_2_stage_arg_parser`'s own `--distilled-lora required=True`, which
  // is what every one of these pipelines selects.
  if (im.recipe.requires_distilled_lora &&
      VideoExtra(params.extras, kLtx2LoraPathExtra).empty()) {
    Fail("the '" + im.pipeline_kind +
         "' pipeline needs a distilled LoRA and none was supplied. Upstream's "
         "`--distilled-lora` is `required=True` on `default_2_stage_arg_parser`, which "
         "this pipeline selects (ltx-pipelines utils/args.py:1123, :1140-1155), and its "
         "second stage is a three-sigma refinement on STAGE_2_DISTILLED_SIGMAS "
         "(utils/constants.py:19-23) that the base weights were never distilled for. "
         "Supply it through the '" +
         std::string(kLtx2LoraPathExtra) +
         "' load extra. Refused rather than rendered, because a distilled schedule on "
         "undistilled weights returns a clip of the right size, frame count and sample rate. "
         "The adapter runs on the phases the recipe's `loras` scope names, which for these "
         "pipelines is stage 2 alone.");
  }
  im.max_phase = ExtraInt(params.extras, kLtx2MaxPhaseExtra, -1);
  if (im.max_phase >= static_cast<int64_t>(im.recipe.phases.size())) {
    Fail("the '" + std::string(kLtx2MaxPhaseExtra) + "' extra is " +
         std::to_string(im.max_phase) + " but the '" + im.pipeline_kind + "'/'" +
         im.model_version + "' recipe has " + std::to_string(im.recipe.phases.size()) +
         " phases");
  }

  // ── the video VAE ─────────────────────────────────────────────────────────
  //
  // REQUIRED, EXCEPT ON AN AUDIO-ONLY RECIPE, and the exception is upstream's
  // shape rather than a convenience: `T2AOneStagePipeline.__init__` constructs a
  // `PromptEncoder`, a `DiffusionStage`, an `AudioDecoder` and a
  // `DurationPredictor` (t2a_one_stage.py:68-107) and never calls
  // `model_paths.video_vae()`. Demanding one would make a text-to-audio load ask
  // for a checkpoint the pipeline cannot use.
  //
  // Keyed on `recipe.audio_only` rather than on the kind STRING, so the next
  // audio-only recipe inherits it instead of silently failing here. Supplying a
  // video VAE anyway is accepted and loaded — it costs the caller memory and
  // nothing else, and refusing it would break a caller who reuses one params
  // object across pipelines.
  if (params.video_vae_path.empty() && !im.recipe.audio_only) Fail("video_vae_path is required");
  if (!params.video_vae_path.empty()) {
    const phase::Scope video_vae_phase("load.video_vae");
    const SafetensorsFile f = SafetensorsFile::Open(params.video_vae_path);
    const nlohmann::json vae_config = Ltx2ReadCheckpointConfig(f);
    im.video_cfg = Ltx2ParseConvVideoDecoderConfig(vae_config, &im.video_kind);
    im.video_weights = Ltx2LoadVaeWeights(f, Ltx2VideoVaeDecoderKeyRules());

    // `ImageConditioner` builds its VideoEncoder from the SAME checkpoint with
    // `VAE_ENCODER_COMFY_KEYS_FILTER` (blocks.py:956-961). It builds it lazily
    // and frees it after the callable returns (:988-993); this engine keeps it
    // resident instead, because the encoder is small next to the DiT and a
    // conditioning image arrives per request. That is a deliberate divergence
    // from upstream's lifecycle and nothing else: same filter, same
    // configurator, same weights.
    if (Ltx2CheckpointHasVideoEncoder(f.Names())) {
      im.video_encoder_cfg = Ltx2ParseConvVideoEncoderConfig(vae_config);
      im.video_encoder_weights = Ltx2LoadVaeWeights(f, Ltx2VideoVaeEncoderKeyRules());
      im.has_video_encoder = true;

      // The encoder's LATENT WIDTH against the DiT's input, asserted rather
      // than assumed. `_prepare_video_encoder_kwargs` reads it from
      // `latent_channels` and NOT from the top-level `out_channels`
      // (model_configurator.py:41-43); a config that got that wrong builds a
      // 3-channel-latent encoder that still runs, still returns a latent, and
      // conditions the DiT on a tensor of the wrong width.
      if (im.video_encoder_cfg.out_channels != im.dit.params.in_channels) {
        Fail("the video VAE encoder emits " +
             std::to_string(im.video_encoder_cfg.out_channels) +
             " latent channels but the DiT takes " +
             std::to_string(im.dit.params.in_channels) +
             ". `_prepare_video_encoder_kwargs` reads this from `vae.latent_channels`, never "
             "from the top-level `vae.out_channels`, which is the DECODER's RGB count "
             "(video_vae/model_configurator.py:41-42, and the flat-layout read at :52).");
      }
      // And its INPUT width, for the same reason in the other direction: the
      // encoder takes RGB, and a config declaring otherwise would silently
      // reinterpret the image planes.
      if (im.video_encoder_cfg.in_channels != 3) {
        Fail("the video VAE encoder declares " +
             std::to_string(im.video_encoder_cfg.in_channels) +
             " input channels; this seam supplies RGB");
      }
    }
  }
  if (!params.video_vae_path.empty() && im.video_cfg.in_channels != im.dit.params.out_channels) {
    Fail("the video VAE takes " + std::to_string(im.video_cfg.in_channels) +
         " latent channels but the DiT emits " + std::to_string(im.dit.params.out_channels));
  }

  // ── the audio VAE + its vocoder ───────────────────────────────────────────
  if (params.audio_vae_path.empty()) Fail("audio_vae_path is required");
  {
    const phase::Scope audio_vae_phase("load.audio_vae");
    const SafetensorsFile f = SafetensorsFile::Open(params.audio_vae_path);
    const nlohmann::json config = Ltx2ReadCheckpointConfig(f);
    im.audio_cfg = Ltx2ParseAudioDecoderConfig(config);
    im.audio_weights = Ltx2LoadVaeWeights(f, Ltx2AudioVaeDecoderKeyRules());
    im.vocoder_cfg = Ltx2ParseVocoderBweConfig(config);
    im.vocoder_weights = Ltx2LoadVaeWeights(f, Ltx2VocoderKeyRules());
    // The ENCODER half, when the checkpoint carries it (#922). Loaded from the
    // same file and the same metadata object, so the encoder and its mel
    // front-end cannot disagree with the decoder about sample rate or mel bins.
    if (Ltx2CheckpointHasAudioEncoder(f.Names())) {
      im.audio_encoder_cfg = Ltx2ParseAudioEncoderConfig(config);
      im.audio_encoder_weights = Ltx2LoadVaeWeights(f, Ltx2AudioVaeEncoderKeyRules());
      im.has_audio_encoder = true;
    }
  }

  // ── the optional latent spatial upsampler (the two-stage recipe's phase 2) ─
  const std::string upsampler_path = VideoExtra(params.extras, "upsampler_path");
  if (!upsampler_path.empty()) {
    const phase::Scope upsampler_phase("load.upsampler");
    const SafetensorsFile f = SafetensorsFile::Open(upsampler_path);
    const nlohmann::json config = Ltx2ReadCheckpointConfig(f);
    im.upsampler_cfg = Ltx2ParseUpsamplerConfig(config);
    im.upsampler_weights = Ltx2LoadVaeWeights(f);
    im.has_upsampler = true;
  }

  // ── the text tower (phase L13) ────────────────────────────────────────────
  //
  // THIS IS THE LAST HOP, and the reason it could be taken here is worth stating
  // once at the code rather than only in the spec: L10 built prompt -> tokens ->
  // tower -> the two caption projections, L9c put the `Embeddings1DConnector` on
  // the render path with the checkpoint's own weights, and neither branch could
  // see the other. L10's refusal said the connector weights were still refused
  // by `Ltx2LoadDitFromSafetensors`; `ltx2_loader.cpp:417` now records in terms
  // that they are NOT unported and are materialized by `Ltx2LoadConnectorWeights`.
  // So the chain closes, and what is left is composition.
  //
  // The order is fixed by data, not taste: the FEATURE VARIANT is resolved from
  // the DiT's transformer config (`Ltx2SelectTextFeatureVariant`,
  // encoder_configurator.py:163-209), so this must run after `dit_config` and
  // after the params adoption that fixes the two stream widths.
  const std::string encoder_config_path =
      VideoExtra(params.extras, kLtx2EncoderConfigPathExtra);
  if (params.encoder_path.empty() && !encoder_config_path.empty()) {
    Fail("the '" + std::string(kLtx2EncoderConfigPathExtra) +
         "' extra names '" + encoder_config_path +
         "' but no encoder_path was supplied, so there is no tower for it to "
         "configure. Refusing rather than ignoring it: an extra that is silently "
         "dropped looks exactly like the feature not working.");
  }
  if (!params.encoder_path.empty()) {
    // W0: ~24 GB of host bf16 at the shipped 12B, and the tower every W4
    // hypothesis is about.
    const phase::Scope text_encoder_phase("load.text_encoder");
    const SafetensorsFile encoder_file = SafetensorsFile::Open(params.encoder_path);

    // 1. THE ASSET PACK. `require_config=false` unconditionally, because the
    //    decision about WHERE the Gemma config comes from is made below with
    //    both candidates in hand; letting the reader throw first would report
    //    "this checkpoint has no metadata" for a caller who supplied the config.
    const Ltx2GemmaAssets assets = Ltx2LoadGemmaAssets(encoder_file, /*require_config=*/false);
    if (assets.has_config && !encoder_config_path.empty()) {
      Fail("the text encoder declares its own __metadata__[\"gemma_config\"] and the '" +
           std::string(kLtx2EncoderConfigPathExtra) + "' extra names '" +
           encoder_config_path +
           "'. Refusing rather than preferring one: layer_types, global_head_dim, "
           "num_global_key_value_heads and attention_k_eq_v each resolve a DIFFERENT "
           "model out of a byte-identical tensor set, so the wrong choice conditions "
           "on a plausible wrong prompt instead of failing. Drop the extra to use the "
           "checkpoint's own config.");
    }
    if (!assets.has_config && encoder_config_path.empty()) {
      Fail("the text encoder '" + params.encoder_path +
           "' declares no __metadata__[\"gemma_config\"], and no '" +
           std::string(kLtx2EncoderConfigPathExtra) +
           "' extra was supplied. The tensor SHAPES cannot supply one: which layers are "
           "full vs sliding (layer_types) decides their head_dim and kv head count, and "
           "attention_k_eq_v decides whether a missing v_proj is an architecture or a "
           "damaged file. This is the shape the shipped vonkaiser text encoder is in — it "
           "carries no __metadata__ at all, which is why upstream's own "
           "GemmaAssets.from_single_file raises on it (gemma_assets.py:110-114). Supply "
           "the config rather than inheriting a default that resolves a different tower.");
    }
    const nlohmann::json gemma_config =
        assets.has_config ? assets.config
                          : ReadJsonFile(kLtx2EncoderConfigPathExtra, encoder_config_path);

    // 2. THE TOKENIZER, out of the `tokenizer_json` TENSOR — a loader assuming a
    //    sibling tokenizer.json fails on every shipped build of this family.
    im.tokenizer = std::make_unique<tok::Tokenizer>(tok::Tokenizer::FromHfJsonBytes(
        std::string_view(reinterpret_cast<const char*>(assets.tokenizer_json.data()),
                         assets.tokenizer_json.size()),
        params.encoder_path + "::tokenizer_json"));
    im.gemma_ids = Ltx2ResolveGemmaSpecialIds(assets, *im.tokenizer);

    // 3. THE TOWER. ~24 GB of host bf16 at the shipped 12B; see the Impl note.
    im.tower = std::make_unique<Ltx2GemmaTower>(
        Ltx2LoadGemmaTowerFromSafetensors(encoder_file, gemma_config));

    // 4. THE TWO CAPTION PROJECTIONS, and the V1/V2 shape they belong to.
    const Ltx2TextEncoderCheckpoint te = Ltx2LoadTextEncoderFromSafetensors(encoder_file);
    im.feature_cfg = Ltx2SelectTextFeatureVariant(
        dit_config.contains("transformer") ? dit_config.at("transformer") : dit_config,
        te.gemma_hidden_size, te.gemma_num_hidden_layers);
    im.caption_projections = Ltx2WidenTextProjectionsToF32(te);

    // 5. THE TWO WIDTHS MUST BE THE DiT's. `Ltx2SelectTextFeatureVariant` reads
    //    them from the SAME transformer config the DiT's cross-attention
    //    dimensions come from, so a disagreement here means the config and the
    //    encoder file describe different models — checked rather than assumed,
    //    because the failure downstream would be a wrong-shaped GEMM inside the
    //    DiT rather than a load refusal.
    if (im.feature_cfg.video_out_features != im.dit.params.cross_attention_dim) {
      Fail("the video caption projection emits " +
           std::to_string(im.feature_cfg.video_out_features) +
           " features but the DiT's cross-attention takes " +
           std::to_string(im.dit.params.cross_attention_dim));
    }
    if (im.feature_cfg.audio_out_features != im.dit.params.audio_cross_attention_dim) {
      Fail("the audio caption projection emits " +
           std::to_string(im.feature_cfg.audio_out_features) +
           " features but the DiT's audio cross-attention takes " +
           std::to_string(im.dit.params.audio_cross_attention_dim) +
           ". LTX-2.5 conditions TWO streams and a V1 config emits only one "
           "(encoder_configurator.py:185-188), so an audio width of 0 here means the "
           "checkpoint's config resolved the pre-2.5 single-projection form.");
    }

    // 6. THE REGISTER TILING, at the width the TOKENIZER will produce. Upstream
    //    asserts the same thing (embeddings_connector.py:144) because the
    //    register table is TILED across the sequence rather than indexed by
    //    which positions were padded. Checked at LOAD rather than on the first
    //    request: it depends on nothing a request supplies.
    if (im.has_connector) {
      const int64_t registers = im.video_connector_cfg.num_learnable_registers;
      if (registers > 0 && kLtx2GemmaTokenizerMaxLength % registers != 0) {
        Fail("the tokenizer pads every prompt to " +
             std::to_string(kLtx2GemmaTokenizerMaxLength) +
             " rows (gemma_assets.py:162), which is not a multiple of the connector's " +
             std::to_string(registers) + " learnable registers");
      }
    }
    im.has_encoder = true;
  }

  // ── conditioning from a FILE (the seam's documented fallback) ─────────────
  //
  // Still served, and still the only conditioning when no tower is loaded. With
  // a tower it becomes the fallback for a request that carries no prompt, which
  // is why the two are not mutually exclusive.
  const std::string audio_embeds_path = VideoExtra(params.extras, kLtx2AudioPromptEmbedsExtra);
  if (params.prompt_embeds_path.empty() != audio_embeds_path.empty()) {
    Fail(
        "LTX-2.5 conditions TWO streams at two widths, so prompt_embeds_path (the video "
        "stream, " +
        std::to_string(im.dit.params.cross_attention_dim) + " wide) and the '" +
        std::string(kLtx2AudioPromptEmbedsExtra) + "' extra (the audio stream, " +
        std::to_string(im.dit.params.audio_cross_attention_dim) +
        " wide) are supplied together or not at all. One of them alone would leave a stream "
        "unconditioned, which renders.");
  }
  if (!params.prompt_embeds_path.empty()) {
    const phase::Scope embeds_phase("load.prompt_embeds");
    im.video_prompt_embeds = ReadF32File("prompt_embeds_path", params.prompt_embeds_path);
    im.audio_prompt_embeds = ReadF32File(kLtx2AudioPromptEmbedsExtra, audio_embeds_path);
    const int64_t vw = im.dit.params.cross_attention_dim;
    const int64_t aw = im.dit.params.audio_cross_attention_dim;
    if (im.video_prompt_embeds.size() % static_cast<size_t>(vw) != 0) {
      Fail("prompt_embeds_path holds " + std::to_string(im.video_prompt_embeds.size()) +
           " floats, which is not a whole number of " + std::to_string(vw) + "-wide rows");
    }
    if (im.audio_prompt_embeds.size() % static_cast<size_t>(aw) != 0) {
      Fail(std::string(kLtx2AudioPromptEmbedsExtra) + " holds " +
           std::to_string(im.audio_prompt_embeds.size()) +
           " floats, which is not a whole number of " + std::to_string(aw) + "-wide rows");
    }
    const int64_t v_rows = static_cast<int64_t>(im.video_prompt_embeds.size()) / vw;
    const int64_t a_rows = static_cast<int64_t>(im.audio_prompt_embeds.size()) / aw;
    if (v_rows != a_rows) {
      // Upstream's two encodings come from ONE tokenization
      // (feature_extractor.py:114-129 projects the SAME hidden states twice), so
      // differing row counts mean two different prompts, and the render would be
      // a picture of one with a soundtrack of the other.
      Fail("prompt_embeds_path has " + std::to_string(v_rows) + " rows but " +
           std::string(kLtx2AudioPromptEmbedsExtra) + " has " + std::to_string(a_rows) +
           "; both projections are built from ONE tokenization and must agree");
    }
    if (v_rows == 0) Fail("the prompt embeds are empty");
    im.prompt_tokens = v_rows;

    // How much of the supplied conditioning is REAL. Everything by default; the
    // extra is what a caller with a tokenizer's mask supplies.
    im.prompt_valid_rows = ExtraInt(params.extras, kLtx2PromptValidRowsExtra, v_rows);
    if (im.prompt_valid_rows < 0 || im.prompt_valid_rows > v_rows) {
      Fail("the '" + std::string(kLtx2PromptValidRowsExtra) + "' extra is " +
           std::to_string(im.prompt_valid_rows) + " but the prompt embeds hold " +
           std::to_string(v_rows) + " rows");
    }
    if (im.has_connector) {
      const int64_t registers = im.video_connector_cfg.num_learnable_registers;
      if (registers > 0 && v_rows % registers != 0) {
        // Upstream asserts this (embeddings_connector.py:144) because the
        // register table is TILED across the sequence rather than indexed by
        // which positions were padded.
        Fail("the prompt embeds hold " + std::to_string(v_rows) +
             " rows, which is not a multiple of the connector's " +
             std::to_string(registers) +
             " learnable registers. The register table is tiled across the sequence "
             "(embeddings_connector.py:144), so upstream asserts the same thing.");
      }
      // The additive mask `_prepare_attention_mask` would produce: 0.0 for a real
      // token, -finfo(f32).max for padding.
      std::vector<float> additive(static_cast<size_t>(v_rows), 0.0f);
      for (int64_t s = im.prompt_valid_rows; s < v_rows; ++s) {
        additive[static_cast<size_t>(s)] = -std::numeric_limits<float>::max();
      }
      // ONE statement of the connector call and its mask contract, shared with
      // the per-request prompt path (`RunConnector`).
      const Ltx2ConnectorEmbeddings encoded =
          RunConnector(dit_file, im.video_connector_cfg, im.audio_connector_cfg,
                       im.video_prompt_embeds, im.audio_prompt_embeds, additive, v_rows);
      im.video_prompt_embeds = encoded.video;
      im.audio_prompt_embeds = encoded.audio;
    }

    // ── the NEGATIVE half (row LTX25-GUIDED-VIDEO, #1092) ──────────────────
    //
    // The same two files for upstream's second encoding. Loaded here, INSIDE the
    // positive block, because a negative pair without a positive one conditions
    // nothing: `prompt_embeds_path` is what a tower-less engine renders from.
    // The two negative files follow the positive pair's own rule — supplied
    // together or not at all — and must agree with it on row count, because the
    // guidance delta subtracts them elementwise.
    const std::string negative_video_path =
        VideoExtra(params.extras, kLtx2NegativePromptEmbedsExtra);
    const std::string negative_audio_path =
        VideoExtra(params.extras, kLtx2NegativeAudioPromptEmbedsExtra);
    if (negative_video_path.empty() != negative_audio_path.empty()) {
      Fail("the '" + std::string(kLtx2NegativePromptEmbedsExtra) + "' and '" +
           std::string(kLtx2NegativeAudioPromptEmbedsExtra) +
           "' extras are supplied together or not at all, for the same reason the positive pair "
           "is: LTX-2.5 conditions TWO streams at two widths and one of them alone would leave a "
           "stream unconditioned on the unconditional forward, which renders.");
    }
    if (!negative_video_path.empty()) {
      im.negative_video_prompt_embeds =
          ReadF32File(kLtx2NegativePromptEmbedsExtra, negative_video_path);
      im.negative_audio_prompt_embeds =
          ReadF32File(kLtx2NegativeAudioPromptEmbedsExtra, negative_audio_path);
      if (static_cast<int64_t>(im.negative_video_prompt_embeds.size()) != v_rows * vw ||
          static_cast<int64_t>(im.negative_audio_prompt_embeds.size()) != a_rows * aw) {
        Fail("the negative prompt embeds hold " +
             std::to_string(im.negative_video_prompt_embeds.size()) + " / " +
             std::to_string(im.negative_audio_prompt_embeds.size()) +
             " floats and the positive pair holds " +
             std::to_string(im.video_prompt_embeds.size()) + " / " +
             std::to_string(im.audio_prompt_embeds.size()) +
             " at widths " + std::to_string(vw) + " / " + std::to_string(aw) +
             ". Upstream encodes `[prompt, negative_prompt]` in ONE call, so the two halves "
             "share a padded width by construction and `(cfg_scale - 1) * (cond - uncond)` "
             "subtracts them elementwise");
      }
      if (im.has_connector) {
        // Through the SAME connector, with the SAME mask. A negative stream that
        // skipped it would be compared against a positive stream that did not,
        // and the delta would be dominated by the connector rather than by the
        // prompt.
        std::vector<float> additive(static_cast<size_t>(v_rows), 0.0f);
        for (int64_t s = im.prompt_valid_rows; s < v_rows; ++s) {
          additive[static_cast<size_t>(s)] = -std::numeric_limits<float>::max();
        }
        const Ltx2ConnectorEmbeddings encoded = RunConnector(
            dit_file, im.video_connector_cfg, im.audio_connector_cfg,
            im.negative_video_prompt_embeds, im.negative_audio_prompt_embeds, additive, v_rows);
        im.negative_video_prompt_embeds = encoded.video;
        im.negative_audio_prompt_embeds = encoded.audio;
      }
    }
  }
  return engine;
}

namespace {

// GENERATED keyframe slots — SERVED as of row LTX25-DFR-PIPELINE (#986), and the
// DFR canvas that drives them.
//
// The refusal this block replaces is recorded rather than deleted, because the
// retirement IS the record. Row LTX25-GENERATED-KEYFRAMES (#920) defined
// `num_generated_keyframes` and refused any positive count, naming ONE blocker:
// the READBACK — `GeneratedKeyframeLayout`, the extraction into
// `generated_keyframes` before the trim, and a standalone single-frame decode of
// each slot. That refusal was CORRECT when it was written and it is now false in
// its first two thirds, which #986 landed:
// `Ltx2ConditionVideoByGeneratedKeyframeSlots` and
// `Ltx2ExtractGeneratedKeyframes` in `ltx2_conditioning.h`, with
// `Ltx2ClearConditioning` extracting BEFORE it trims.
//
// The third piece — the standalone decode — is still owed, and it is NOT a
// blocker for anything served here. It belongs to a surface that would hand slot
// PIXELS back to a caller, and neither DFR nor this engine has one: DFR keeps
// its slots in latent space from end to end, upsampling them
// (dfr_pipeline.py:348) and feeding them back as `initial_keyframes` (:364).
// Tracked as owed under #986.
//
// The refusal carried a gated tripwire aimed at exactly this change —
// `ABSENT HERE: GeneratedKeyframe, generated_keyframe`, re-derived by
// `test_ltx2_video` against `ltx2_conditioning.h`'s declarations. It fired, as
// its own spec (`.agents/specs/ltx25-generated-keyframes.md` section 4a)
// predicted in terms. The assertion was retired with the refusal it described,
// never widened.
//
// Deliberately a second anonymous namespace rather than an addition to the one
// at the top of this file: the READER ANCHORS comment above `kKnownLoadExtras`
// carries derived LINE NUMBERS into this file and is gated by
// `test_ltx2_video`, so a definition inserted up there would move every anchor
// under it and break that gate for a reason that has nothing to do with this
// row. Everything here sits below the last anchored line.
//
// This resolves the request BEFORE any arm is selected, so the FP8, NVFP4 and
// bf16 arms cannot reach the unported machinery by different routes — there is
// one answer for the family, not one per arm.
// `evenly_spaced_keyframe_positions` (ltx-pipelines/utils/helpers.py:370-381):
// `linspace(0, num_frames - 1, n + 2).round().to(int64)[1:-1]` — interior
// positions with BOTH ENDPOINTS DROPPED.
//
// THE ENDPOINT DROP IS THE WHOLE FUNCTION. `linspace` puts its first sample on
// frame 0 and its last on `num_frames - 1`, and both are excluded: frame 0
// already spans a single pixel frame under causal encoding, so a slot there buys
// nothing, and the terminal frame is the clip's own end. A port that kept them
// would place `n + 2` slots, cost two extra latent frames of tokens, and render.
//
// `torch.linspace` divides by `steps - 1` in f64 and `.round()` is half-to-even,
// mirrored here for the same reason `Ltx2DfrSlotInitialsFromVideo` mirrors it.
std::vector<int64_t> EvenlySpacedKeyframePositions(int64_t num_keyframes, int64_t num_frames) {
  if (num_keyframes < 0) {
    Fail("the '" + std::string(kLtx2GeneratedKeyframesExtra) + "' extra is " +
         std::to_string(num_keyframes) +
         ", and num_keyframes must be non-negative — upstream's own refusal, raised by "
         "`evenly_spaced_keyframe_positions` (ltx-pipelines/utils/helpers.py:372-373) before "
         "the checkpoint is consulted. Use 0 to turn generated keyframes off, which is "
         "upstream's default (utils/args.py:836).");
  }
  if (num_keyframes == 0) return {};
  if (num_frames < num_keyframes + 2) {
    Fail("generated keyframes need at least num_keyframes + 2 target frames, got "
         "num_keyframes=" +
         std::to_string(num_keyframes) + ", num_frames=" + std::to_string(num_frames) +
         " (ltx-pipelines/utils/helpers.py:374-378). Each slot is an INTERIOR position, so the "
         "two endpoints are not available to it.");
  }
  const int64_t steps = num_keyframes + 2;
  std::vector<int64_t> positions;
  positions.reserve(static_cast<size_t>(num_keyframes));
  for (int64_t i = 1; i + 1 < steps; ++i) {
    const double value = static_cast<double>(num_frames - 1) * static_cast<double>(i) /
                         static_cast<double>(steps - 1);
    positions.push_back(static_cast<int64_t>(std::nearbyint(value)));
  }
  return positions;
}

void CheckGeneratedKeyframes(const std::map<std::string, std::string>& extras,
                             const std::string& pipeline_kind) {
  if (VideoExtra(extras, kLtx2GeneratedKeyframesExtra).empty()) return;
  const int64_t count = ExtraInt(extras, kLtx2GeneratedKeyframesExtra, 0);

  // DFR OWNS ITS OWN SLOT POSITIONS. `DFRPipeline.__call__` takes no
  // `generated_keyframes` argument at all: the positions come from
  // `resolve_canvas` (dfr_pipeline.py:314), one per x8-border segment boundary,
  // and the whole pipeline — the tile ranges, the carry-forward bag, the
  // anchor seams — is built on that grid. Honouring an override here would
  // detach the slots from the canvas that indexes them, and every later stage
  // would still run.
  if (pipeline_kind == "dfr" && count != 0) {
    Fail("the '" + std::string(kLtx2GeneratedKeyframesExtra) +
         "' extra is not accepted on the 'dfr' pipeline, which chooses its own keyframe "
         "positions. `DFRPipeline.__call__` takes no such argument and its CLI exposes no "
         "`--num-generated-keyframes` (ltx-pipelines/dfr_pipeline.py:268-283, :565-591): the "
         "slots sit on the segment grid `resolve_canvas` returns (:314, dfr_layout.py:60-81), "
         "one per x8 border, and the tile ranges and carry-forward bag are indexed by that same "
         "grid. Refused rather than honoured, because an override would leave the slots and the "
         "canvas describing different frames and the render would still finish. Use "
         "'pipeline_kind' 'distilled_two_stage' or 'one_stage' to place slots by count.");
  }

  // ZERO IS UPSTREAM'S DEFAULT, AND IT IS OFF. `args.py:836` is `default=0` and
  // `has_generated_keyframes` (utils/helpers.py:384-391) reads 0 as "no slots
  // requested". A caller that plumbs the default through must get a render.
  // Refusing on the mere presence of the key is one line shorter and wrong, and
  // it stays wrong now that the arm is served: the DFR refusal above is keyed on
  // `count != 0` for the same reason.
  if (count == 0) return;

  // A MALFORMED REQUEST AND AN UNPORTED ARM ARE DIFFERENT ANSWERS, and upstream
  // gives this one first: `evenly_spaced_keyframe_positions` raises
  // "num_keyframes must be non-negative" (utils/helpers.py:372-373) before
  // anything looks at the checkpoint. Resolved through the shared helper so the
  // request surface and the render path cannot disagree about what is legal.
  (void)EvenlySpacedKeyframePositions(count, /*num_frames=*/count + 2);
}

// `DiffusionStage.assert_generated_keyframes_supported`
// (ltx-pipelines/utils/blocks.py:405-419), called from a pipeline's `__call__`
// preamble (dfr_pipeline.py:315) BEFORE any weight is built.
//
// It reads the DECLARED config flag and refuses rather than degrading, and
// upstream states the reason at keyframe_slots.py:9-12: on a checkpoint without
// the marker "the slots would be denoised as unmarked tokens and the extra
// compute would be wasted". Each slot costs one latent frame of tokens to buy
// ONE pixel frame, so a silent degradation is not a smaller feature — it is the
// same bill for nothing.
//
// OURS RESOLVES TO SHAPES, NOT TO THE DECLARATION, and the difference is
// recorded because #902 asked about exactly it. `Ltx2AdoptDeclaredDitParams`
// mirrors `LTXModel.supports_keyframes_abs_pos_embedding`
// (model/transformer/model.py:166-173), which reads the MATERIALIZED tensor, so
// a checkpoint that declares the flag and ships no keyframe tensor resolves
// FALSE here and refuses. Upstream's admission gate reads the declaration alone
// and would let that request through, then reach `embedding.to(dtype=...)` on a
// meta tensor. Ours is the safer of the two and the divergence is deliberate.
void AssertGeneratedKeyframesSupported(bool has_embedding, const std::string& dit_path) {
  if (has_embedding) return;
  Fail("generated keyframe slots were requested, but the DiT at '" + dit_path +
       "' carries no keyframe absolute-position embedding, so it has no trained marker to put "
       "on them. Upstream refuses the same request at the same point "
       "(`assert_generated_keyframes_supported`, ltx-pipelines/utils/blocks.py:405-419, called "
       "from the pipeline preamble before any weight is built) and says why at "
       "ltx-core/conditioning/types/keyframe_slots.py:9-12: without the marker the slots are "
       "denoised as ORDINARY tokens while still costing one latent frame of tokens each to buy "
       "one pixel frame. Degrading silently would charge the full bill for nothing. "
       "WHAT IS *NOT* THE REASON: the marker is not unported. `keyframes_abs_pos_embedding` "
       "landed under row LTX25-KEYFRAMES-ABS-POS (#658) and is applied on EVERY render, because "
       "`_first_frame_keyframes_mask` (ltx_core/tools.py:184-196) marks the target's first "
       "latent frame unconditionally. What is absent is this CHECKPOINT's tensor. "
       "This engine resolves the capability from the materialized tensor rather than from the "
       "declared config flag, mirroring `LTXModel.supports_keyframes_abs_pos_embedding` "
       "(model/transformer/model.py:166-173) rather than the pipeline-side gate, so a "
       "checkpoint that DECLARES the flag and ships no tensor is refused here and would be "
       "admitted upstream (#902). Supply a generated-keyframe checkpoint, or drop the request.");
}

// ── the guiders (row LTX25-GUIDED-VIDEO, #1092) ────────────────────────────

// `--*-stg-blocks`, `nargs="*"` (utils/args.py:979-985, :1039-1045). An extra
// that is PRESENT and empty is upstream's empty list — "perturb nothing" — and
// stays distinct from an ABSENT extra, which takes the params table's own value.
// Collapsing the two would make `video_stg_blocks=` silently mean block 28.
void ApplyStgBlocksExtra(const std::map<std::string, std::string>& extras, const char* key,
                         std::vector<int64_t>* blocks) {
  const auto at = extras.find(key);
  if (at == extras.end()) return;
  blocks->clear();
  const std::string& raw = at->second;
  for (size_t i = 0; i < raw.size();) {
    const size_t comma = raw.find(',', i);
    const std::string token = raw.substr(i, comma == std::string::npos ? comma : comma - i);
    if (!token.empty()) {
      try {
        blocks->push_back(std::stoll(token));
      } catch (const std::exception&) {
        Fail("'" + std::string(key) + "' holds '" + token +
             "', which is not an integer block index");
      }
    }
    if (comma == std::string::npos) break;
    i = comma + 1;
  }
}

// One CLI flag each, from `default_1_stage_arg_parser` (utils/args.py:947-1066:
// the video row's six flags open at :948 and the audio row's at :1008). Each extra overrides ONE
// field of the phase's own resolved guider, which is what one flag does.
//
// REFUSED WHOLESALE on a phase that fixes its guidance. `allow_guidance_override
// = false` is set by the distilled two-stage and retake recipes
// (ltx2_recipes.py:125-158, retake.py:53) whose guidance is distilled INTO the
// weights, and until this row nothing read it. Honouring an override there would
// sample a trajectory the weights were never trained for — the same argument
// `fixed_num_inference_steps` already makes about the schedule, and the same
// reason it is a refusal rather than a silent clamp.
void ApplyGuidanceOverrides(const std::map<std::string, std::string>& extras,
                            const Ltx2PhaseRecipe& phase, Ltx2MultiModalGuiderParams* video,
                            Ltx2MultiModalGuiderParams* audio) {
  static const char* const kVideoKeys[] = {
      kLtx2VideoCfgScaleExtra, kLtx2VideoStgScaleExtra,  kLtx2VideoRescaleScaleExtra,
      kLtx2VideoSkipStepExtra, kLtx2VideoStgBlocksExtra, kLtx2A2vGuidanceScaleExtra,
      kLtx2AudioCfgScaleExtra, kLtx2AudioStgScaleExtra,  kLtx2AudioRescaleScaleExtra,
      kLtx2AudioSkipStepExtra, kLtx2AudioStgBlocksExtra, kLtx2V2aGuidanceScaleExtra};
  if (!phase.allow_guidance_override) {
    for (const char* key : kVideoKeys) {
      if (extras.find(key) == extras.end()) continue;
      Fail("phase '" + phase.name +
           "' fixes its own guidance, so the '" + std::string(key) +
           "' extra is refused rather than applied. This recipe's scales are distilled INTO the "
           "weights (ltx2_recipes.py:125-158), and a render that honoured the override would "
           "sample a trajectory they were never trained for.");
    }
    return;
  }
  // IGNORED, not refused, on a phase whose denoiser takes no params. This is
  // `SimpleDenoiser` (utils/denoisers.py:3) and the a2vid stage 2 is the one
  // phase in the table that reaches it: the flags exist on that pipeline's
  // parser (a2vid_two_stage.py:311 -> utils/args.py:947-1006) and they reach
  // stage 1's guider alone (`:233-236`), because stage 2 constructs
  // `SimpleDenoiser(v_context_p, a_context_p)` (`:278`). Applying them here
  // instead would switch on a guidance pass upstream's stage 2 does not run —
  // and it would do it invisibly, since an extra forward changes no output
  // shape, frame count or sample rate.
  if (phase.denoiser == Ltx2PhaseDenoiser::kSimple) return;
  video->cfg_scale = ExtraDouble(extras, kLtx2VideoCfgScaleExtra, video->cfg_scale);
  video->stg_scale = ExtraDouble(extras, kLtx2VideoStgScaleExtra, video->stg_scale);
  video->rescale_scale = ExtraDouble(extras, kLtx2VideoRescaleScaleExtra, video->rescale_scale);
  video->modality_scale = ExtraDouble(extras, kLtx2A2vGuidanceScaleExtra, video->modality_scale);
  video->skip_step = ExtraInt(extras, kLtx2VideoSkipStepExtra, video->skip_step);
  ApplyStgBlocksExtra(extras, kLtx2VideoStgBlocksExtra, &video->stg_blocks);

  audio->cfg_scale = ExtraDouble(extras, kLtx2AudioCfgScaleExtra, audio->cfg_scale);
  audio->stg_scale = ExtraDouble(extras, kLtx2AudioStgScaleExtra, audio->stg_scale);
  audio->rescale_scale = ExtraDouble(extras, kLtx2AudioRescaleScaleExtra, audio->rescale_scale);
  audio->modality_scale = ExtraDouble(extras, kLtx2V2aGuidanceScaleExtra, audio->modality_scale);
  audio->skip_step = ExtraInt(extras, kLtx2AudioSkipStepExtra, audio->skip_step);
  ApplyStgBlocksExtra(extras, kLtx2AudioStgBlocksExtra, &audio->stg_blocks);

  const auto check_skip = [](const char* key, int64_t value) {
    if (value >= 0) return;
    Fail("'" + std::string(key) + "' is " + std::to_string(value) +
         "; `should_skip_step` is `step % (skip_step + 1)` (guiders.py:287-291) and a negative "
         "value would take the modulus of a non-positive divisor");
  };
  check_skip(kLtx2VideoSkipStepExtra, video->skip_step);
  check_skip(kLtx2AudioSkipStepExtra, audio->skip_step);
  // AN EMPTY LIST IS NOT REFUSED, and this function refused it until 2026-08-17.
  //
  // The refusal read: an empty `stg_blocks` beside a non-zero STG scale is a
  // perturbed pass identical to the conditional one, so it is a wasted forward
  // and a guidance term of exactly zero. Every clause of that is true and none
  // of it makes the configuration illegal upstream, which is the only question
  // a mirror gets to ask. Measured at Lightricks/LTX-2 `fd4ded7f`:
  //
  //   - `packages/ltx-pipelines/docs/multimodal-guidance.md:13` documents it as
  //     THE way to turn STG off: "Set to `[]` to disable STG", in the same table
  //     and the same idiom as `stg_scale` -> 0.0 and `cfg_scale` -> 1.0.
  //   - `MultiModalGuiderParams.stg_blocks` DEFAULTS to `[]`
  //     (guiders.py:204, `field(default_factory=list)`).
  //   - `--video-stg-blocks` / `--audio-stg-blocks` are `nargs="*"`
  //     (args.py:979-985, :1039-1045, :1107-1113), so the flag with zero values
  //     parses to `[]`. `nargs="+"` was the one-character way to forbid it.
  //   - `LTX_2_3_HQ_PARAMS` SHIPS `stg_blocks=[]` on both modalities
  //     (constants.py:105, :113).
  //   - There is no validation of `stg_blocks` anywhere in that tree: no
  //     emptiness check, no length check, no range check against the block
  //     count.
  //
  // Upstream's semantics are unambiguous and are the reason `[]` is meaningful:
  // `blocks=None` means EVERY block and `blocks=[]` means NO block
  // (perturbations.py:26-33). The empty list is how a caller says the second
  // thing, and `ApplyStgBlocksExtra` above exists to keep PRESENT-and-empty
  // distinct from ABSENT for exactly that reason. Refusing it here made that
  // distinction unreachable.
  //
  // WHAT IS STILL REFUSED, one layer down in `Ltx2GuidedDenoise`: a list that
  // NAMES blocks and reaches none of them, e.g. `[28]` on a two-block DiT. That
  // is a local condition rather than an upstream one — upstream only ever runs
  // 48-block checkpoints and this port runs reduced ones — and it is a mismatch
  // between a request and a checkpoint rather than an expressed intent.
}

// Everything step 0 of phase 0 produced, for the gate that decides WHICH SPACE
// each arm was combined in. Derived at the call from what the seam returned, so
// a mutation to any arm moves a recorded field rather than leaving a comment
// that compiles.
void RecordFirstGuidedStep(Ltx2ConditioningTrace* trace, const Ltx2GuidedDenoiseResult& guided,
                           const std::vector<float>& latent,
                           const std::vector<float>& timesteps, double sigma,
                           const std::vector<float>& stepper_input) {
  const auto slot = [](Ltx2DenoisePass pass) { return static_cast<size_t>(pass); };
  trace->video_guided = true;
  trace->video_cond_forwards = guided.pass_ran[slot(Ltx2DenoisePass::kCond)] ? 1 : 0;
  trace->video_uncond_forwards = guided.pass_ran[slot(Ltx2DenoisePass::kUncond)] ? 1 : 0;
  trace->video_perturbed_forwards = guided.pass_ran[slot(Ltx2DenoisePass::kPerturbed)] ? 1 : 0;
  trace->video_modality_forwards = guided.pass_ran[slot(Ltx2DenoisePass::kModality)] ? 1 : 0;
  trace->video_perturbed_blocks = guided.perturbed_video_blocks;
  trace->video_audio_perturbed_blocks = guided.perturbed_audio_blocks;
  trace->video_modality_skipped_a2v = guided.modality_pass_skipped_a2v;
  trace->video_modality_skipped_v2a = guided.modality_pass_skipped_v2a;
  trace->video_first_latent = latent;
  trace->video_first_timesteps = timesteps;
  trace->video_first_cond = guided.video_pass[slot(Ltx2DenoisePass::kCond)];
  trace->video_first_cond_velocity = guided.video_pass_velocity[slot(Ltx2DenoisePass::kCond)];
  trace->video_first_uncond = guided.video_pass[slot(Ltx2DenoisePass::kUncond)];
  trace->video_first_uncond_velocity = guided.video_pass_velocity[slot(Ltx2DenoisePass::kUncond)];
  trace->video_first_perturbed = guided.video_pass[slot(Ltx2DenoisePass::kPerturbed)];
  trace->video_first_perturbed_velocity =
      guided.video_pass_velocity[slot(Ltx2DenoisePass::kPerturbed)];
  trace->video_first_modality = guided.video_pass[slot(Ltx2DenoisePass::kModality)];
  trace->video_first_modality_velocity =
      guided.video_pass_velocity[slot(Ltx2DenoisePass::kModality)];
  trace->video_first_denoised = guided.video_denoised;
  trace->video_first_stepper_input = stepper_input;
  trace->video_first_sigma = sigma;
}

}  // namespace

VideoResult Ltx2VideoEngine::Generate(const VideoGenParams& gen) {
  Impl& im = *impl_;
  std::lock_guard<std::mutex> guard(im.mutex);

  // ── W0: this render's slice of the timeline (#1010) ───────────────────────
  //
  // The counter is a process static rather than a member: the table is a PROCESS
  // timeline (the load is render 0 and belongs to whichever render first flushes
  // it), and two engines in one process interleaving their renders is a
  // measurement shape nobody takes, not a case to encode.
  static std::atomic<int64_t> render_counter{0};
  phase::PhaseLog::Instance().SetRender(render_counter.fetch_add(1) + 1);
  phase::Scope generate_span("generate", /*span=*/true);
  const std::string phase_device =
      im.on_device ? std::string(vt::DeviceTypeName(im.device.type)) : std::string("cpu");

  if (gen.output_dir.empty()) Fail("output_dir is required");
  phase::Scope setup_phase("generate.setup");
  for (const auto& kv : gen.extras) {
    // The per-generation extras this family DEFINES, and the list is the one
    // below rather than this sentence: `image_crf` (row LTX25-IMAGE-COND), the
    // three audio-to-video knobs (row LTX25-A2V-AUDIO-INPUT, #922),
    // `num_generated_keyframes` (defined by row LTX25-GENERATED-KEYFRAMES #920,
    // SERVED by row LTX25-DFR-PIPELINE #986) and `temporal_upsample_rounds`
    // (#986). DEFINED is still not SERVED — the last one is defined so that its
    // own refusal can name the missing loop, exactly as `CheckUnservedExtras`
    // does on the load side (#611). Everything OUTSIDE the list is refused
    // rather than ignored, for the reason `CheckKnownExtras` gives for the load
    // side: a mistyped knob that is silently dropped renders the DEFAULT and
    // looks like the feature not working.
    const bool known = kv.first == kLtx2ImageCrfExtra || kv.first == kLtx2AudioPathExtra ||
                       kv.first == kLtx2AudioStartTimeExtra ||
                       kv.first == kLtx2AudioMaxDurationExtra ||
                       kv.first == kLtx2GeneratedKeyframesExtra ||
                       kv.first == kLtx2TemporalRoundsExtra ||
                       kv.first == kLtx2RetakeStartTimeExtra ||
                       kv.first == kLtx2RetakeEndTimeExtra ||
                       kv.first == kLtx2RetakeFrameRateExtra ||
                       kv.first == kLtx2RegenerateVideoExtra ||
                       kv.first == kLtx2RegenerateAudioExtra ||
                       kv.first == kLtx2NegativePromptExtra ||
                       kv.first == kLtx2AudioCfgScaleExtra ||
                       kv.first == kLtx2AudioStgScaleExtra ||
                       kv.first == kLtx2AudioRescaleScaleExtra ||
                       kv.first == kLtx2AudioSkipStepExtra ||
                       kv.first == kLtx2AudioStgBlocksExtra ||
                       // The VIDEO guider's row (row LTX25-GUIDED-VIDEO, #1092),
                       // from the same parser as the audio row above
                       // (utils/args.py:947-1066).
                       kv.first == kLtx2VideoCfgScaleExtra ||
                       kv.first == kLtx2VideoStgScaleExtra ||
                       kv.first == kLtx2VideoRescaleScaleExtra ||
                       kv.first == kLtx2VideoSkipStepExtra ||
                       kv.first == kLtx2VideoStgBlocksExtra ||
                       kv.first == kLtx2A2vGuidanceScaleExtra ||
                       kv.first == kLtx2V2aGuidanceScaleExtra;
    if (!known) {
      Fail("unknown per-generation extra '" + kv.first + "'. This family defines: " +
           std::string(kLtx2ImageCrfExtra) + ", " + kLtx2AudioPathExtra + ", " +
           kLtx2AudioStartTimeExtra + ", " + kLtx2AudioMaxDurationExtra + ", " +
           kLtx2GeneratedKeyframesExtra + ", " + kLtx2TemporalRoundsExtra + ", " +
           kLtx2RetakeStartTimeExtra + ", " + kLtx2RetakeEndTimeExtra + ", " +
           kLtx2RetakeFrameRateExtra + ", " + kLtx2RegenerateVideoExtra + ", " +
           kLtx2RegenerateAudioExtra + ", " + kLtx2NegativePromptExtra + ", " +
           kLtx2AudioCfgScaleExtra + ", " + kLtx2AudioStgScaleExtra + ", " +
           kLtx2AudioRescaleScaleExtra + ", " + kLtx2AudioSkipStepExtra + ", " +
           kLtx2AudioStgBlocksExtra + ", " + kLtx2VideoCfgScaleExtra + ", " +
           kLtx2VideoStgScaleExtra + ", " + kLtx2VideoRescaleScaleExtra + ", " +
           kLtx2VideoSkipStepExtra + ", " + kLtx2VideoStgBlocksExtra + ", " +
           kLtx2A2vGuidanceScaleExtra + ", " + kLtx2V2aGuidanceScaleExtra);
    }
  }
  // ── the knobs that belong to ONE pipeline (#1005, corrected by #1092) ─────
  //
  // `pipeline_kind` is a LOAD extra, so which pipeline runs is settled before a
  // request arrives and this is a decidable question rather than a guess.
  //
  // WHAT #1092 CORRECTED, and why the old list was defensible until it was not.
  // Row LTX25-T2A-ONE-STAGE refused `negative_prompt` and the five `audio_*`
  // guider knobs on ANY non-t2a engine, reasoning that "no other pipeline
  // `__call__` upstream takes a guider argument at all". That sentence was
  // FALSE about upstream and TRUE about this port. Upstream's
  // `default_1_stage_arg_parser` carries `--negative-prompt`
  // (utils/args.py:937-946) and the whole audio guider row
  // (`:1011-1075`) alongside the video one, and `TI2VidOneStagePipeline`
  // consumes both through `audio_guider_params` (ti2vid_one_stage.py:215-218).
  // What made the refusal harmless was that NOTHING HERE READ THEM on a joint
  // render — the video denoise loop was unguided. Row LTX25-GUIDED-VIDEO makes
  // them live, so the refusal would now reject a flag upstream serves.
  //
  // The guard therefore keeps one direction and drops the other: the knobs that
  // describe a PICTURE are refused on a text-to-audio engine, which produces
  // none. `T2AOneStagePipeline.__call__` takes a prompt, a negative prompt, a
  // seed, a frame rate, a step count, the audio guider and a frame count
  // (t2a_one_stage.py:109-122) and nothing else.
  {
    const char* const kNotOnT2a[] = {kLtx2ImageCrfExtra,        kLtx2AudioPathExtra,
                                     kLtx2AudioStartTimeExtra,  kLtx2AudioMaxDurationExtra,
                                     kLtx2GeneratedKeyframesExtra, kLtx2TemporalRoundsExtra,
                                     kLtx2RetakeStartTimeExtra, kLtx2RetakeEndTimeExtra,
                                     kLtx2RetakeFrameRateExtra, kLtx2RegenerateVideoExtra,
                                     kLtx2RegenerateAudioExtra,
                                     // The VIDEO guider's own row: there is no
                                     // video stream to guide, and upstream's t2a
                                     // parser exposes none of them
                                     // (utils/args.py:1083-1119).
                                     kLtx2VideoCfgScaleExtra,   kLtx2VideoStgScaleExtra,
                                     kLtx2VideoRescaleScaleExtra, kLtx2VideoSkipStepExtra,
                                     kLtx2VideoStgBlocksExtra,  kLtx2A2vGuidanceScaleExtra,
                                     kLtx2V2aGuidanceScaleExtra};
    for (const char* key : kNotOnT2a) {
      if (im.recipe.audio_only && !VideoExtra(gen.extras, key).empty()) {
        Fail("the '" + std::string(key) +
             "' extra has no meaning on a text-to-audio render, which produces no picture at "
             "all. `T2AOneStagePipeline.__call__` takes a prompt, a negative prompt, a seed, a "
             "frame rate, a step count, the audio guider and a frame count "
             "(t2a_one_stage.py:109-122) and nothing else. Refused rather than ignored");
      }
    }
  }
  // The two audio WINDOW knobs only mean something alongside a file. Accepting
  // one on its own would silently do nothing, which is the defect the whole
  // extras surface refuses by name elsewhere.
  {
    const std::string path = VideoExtra(gen.extras, kLtx2AudioPathExtra);
    for (const char* dependent : {kLtx2AudioStartTimeExtra, kLtx2AudioMaxDurationExtra}) {
      if (path.empty() && !VideoExtra(gen.extras, dependent).empty()) {
        Fail("the '" + std::string(dependent) + "' extra was supplied without '" +
             std::string(kLtx2AudioPathExtra) +
             "', so there is no audio for it to window. Refused rather than ignored");
      }
    }
  }
  CheckGeneratedKeyframes(gen.extras, im.pipeline_kind);

  // ── DFR's temporal x2/x4 rounds — DEFINED, refused above 0 (#986) ──────────
  //
  // `temporal_upsample_rounds` (dfr_pipeline.py:277, :584-590), `choices=(0, 1,
  // 2)`, `default=0`. Two answers, and giving the same one to both would be
  // wrong: a value outside the set is a MALFORMED REQUEST and upstream refuses
  // it first (:284-285), before the pipeline looks at anything; a value INSIDE
  // the set is a legal request for a loop this port does not have.
  {
    const std::string raw = VideoExtra(gen.extras, kLtx2TemporalRoundsExtra);
    if (!raw.empty()) {
      const int64_t rounds = ExtraInt(gen.extras, kLtx2TemporalRoundsExtra, 0);
      if (rounds < 0 || rounds > kLtx2DfrMaxTemporalRounds) {
        Fail("the '" + std::string(kLtx2TemporalRoundsExtra) + "' extra is " +
             std::to_string(rounds) +
             "; upstream's own refusal is `temporal_upsample_rounds must be 0, 1, or 2` "
             "(ltx-pipelines/dfr_pipeline.py:284-285), raised at the top of `__call__` before "
             "any work is paid for.");
      }
      // DELIBERATELY NOT GUARDED ON `pipeline_kind` FIRST. A "this knob only
      // means something on the dfr pipeline" refusal would be true, and it would
      // be UNREACHABLE: the rounds loop is unported on every pipeline including
      // `dfr`, so the check below already answers every caller and the
      // pipeline-specific branch could never fire. That is the unselected-branch
      // shape `.agents/reachability.md` enumerates, and writing it now would
      // land a refusal nothing can trip. It belongs to whichever row serves the
      // rounds, and the message below says the knob is DFR's so a caller on
      // another pipeline still learns it.
      if (rounds != 0) {
        Fail(
            "DFR's temporal refinement rounds are not served. This knob is `DFRPipeline`'s alone "
            "(ltx-pipelines/dfr_pipeline.py:277); no other pipeline `__call__` takes one. "
            "WHAT IS MISSING IS THE ROUNDS "
            "LOOP (ltx-pipelines/dfr_pipeline.py:402-529), not the upsampler it calls. Each "
            "round temporally x2-upsamples the video latent (:407), doubles both the playback "
            "and the conditioning fps under a 60 fps cap (:409, :414, :74-78), re-tiles the "
            "canvas into `2**round` keyframe-seam windows (:415), invents mid-segment slots per "
            "tile (:470-478), denoises each tile with an ancestral Euler step at eta 0.5 and a "
            "PER-TILE noise seed (:495-499), stitches the tiles back (:508) and merges the "
            "denoised slots into the next round's anchor bag (:527-529). "
            "WHAT IS *NOT* THE REASON, and each of these was re-derived at this tree rather "
            "than inherited. FIRST, NOT the temporal x2 latent upsampler: it is ported and "
            "gated against executed upstream at reduced dimensions (row "
            "LTX25-TEMPORAL-UPSAMPLER, `.agents/specs/ltx25-temporal-upsampler.md`), including "
            "`PixelShuffle1d` and the first-frame drop (model/upsampler/model.py:68-71, "
            "109-113), and `Ltx2ParseUpsamplerConfig` already reads `temporal_upsample` off a "
            "checkpoint config. SECOND, NOT the canvas layout: `resolve_canvas`, `tile_ranges`, "
            "`stitch_tile_latents` and the carry-forward merge are ported and gated in this "
            "same row (`ltx2_dfr.h`, `test_ltx2_dfr`), so the tile geometry every round needs "
            "already exists and is checked against upstream's own return values. THIRD, NOT the "
            "generated keyframe slots: they are SERVED here, which is what the base and detail "
            "stages run on. What has no local counterpart is the DENOISE PASS AS A CALLABLE — "
            "upstream's rounds loop invokes the same `DiffusionStage.__call__` the two stages "
            "use, per tile, with its own sigmas, stepper and seed, and this engine's denoise is "
            "written inline in one per-phase loop with no seam a tile can enter through. "
            "AND ONE THING IS MISSING THAT NO CODE CAN SUPPLY: the checkpoint. "
            "`ltx-2.5-latent-temporal-upscaler-x2-bf16-1.0.safetensors` "
            "(ltx-pipelines/docs/pipelines.md:176) is NOT on the NAS — the "
            "`latent_upscale_models/` directory holds the SPATIAL upscaler and nothing else, "
            "re-verified 2026-08-16 — so even a complete loop would have no real weights to "
            "run. Use 0, which is upstream's default and the served path. "
            "Tracked as owed by issue #986.");
      }
    }
  }

  // ── RETAKE: the request knobs (#924) ──────────────────────────────────────
  //
  // `RetakePipeline.__call__` (retake.py:151-329). Parsed here, at the top of
  // `Generate`, because upstream refuses an inverted window as its FIRST
  // statement (:211-212) and its CLI refuses the source geometry before the
  // pipeline is even constructed (:340-353) — both before any model work is
  // paid for.
  const bool wants_retake = !VideoExtra(gen.extras, kLtx2RetakeStartTimeExtra).empty() ||
                            !VideoExtra(gen.extras, kLtx2RetakeEndTimeExtra).empty();
  double retake_start = 0.0, retake_end = 0.0, retake_fps = 0.0;
  bool regenerate_video = true, regenerate_audio = true;
  Ltx2RetakeSourceGeometry retake_source;
  if (wants_retake) {
    // Every retake knob is meaningless without the others, and a knob that
    // silently does nothing is the defect this whole surface refuses by name —
    // the precedent is the audio window pair immediately below.
    for (const char* required :
         {kLtx2RetakeStartTimeExtra, kLtx2RetakeEndTimeExtra, kLtx2RetakeFrameRateExtra}) {
      if (VideoExtra(gen.extras, required).empty()) {
        Fail("this is a retake request and the '" + std::string(required) +
             "' extra is missing. A retake needs the window's start and end in seconds "
             "(retake.py:155-156) and the source folder's frame rate, which has no container "
             "to be read from (media_io/decode.py:213-215) and which the whole temporal mask "
             "is divided by (noise_mask_cond.py:35) — defaulting any of the three would "
             "regenerate the wrong seconds and still render");
      }
    }
    if (gen.ref_video_dir.empty()) {
      Fail("this is a retake request and no source clip was supplied. Set `ref_video` to a "
           "DIRECTORY of frame_%06d.ppm. A container file is REFUSED rather than read: "
           "upstream opens one with PyAV (media_io/decode.py:226) and no demuxer is vendored "
           "here. That is upstream's own second ingestion arm, not a substitute — it carries "
           "a frame-folder path for exactly the no-container case (utils/helpers.py:197-220)");
    }
    if (!VideoExtra(gen.extras, kLtx2AudioPathExtra).empty()) {
      Fail("a retake request cannot also carry '" + std::string(kLtx2AudioPathExtra) +
           "'. Retake takes its audio from the SOURCE file and from nowhere else "
           "(retake.py:250-256), so accepting both would silently pick one of two "
           "soundtracks and render");
    }
    retake_start = ExtraDouble(gen.extras, kLtx2RetakeStartTimeExtra, 0.0);
    retake_end = ExtraDouble(gen.extras, kLtx2RetakeEndTimeExtra, 0.0);
    retake_fps = ExtraDouble(gen.extras, kLtx2RetakeFrameRateExtra, 0.0);
    Ltx2RetakeAssertWindow(retake_start, retake_end);
    if (retake_fps <= 0.0) {
      Fail("the '" + std::string(kLtx2RetakeFrameRateExtra) + "' extra is " +
           std::to_string(retake_fps) + ", and a frame rate must be positive");
    }
    // `regenerate_video` / `regenerate_audio` (retake.py:164-165), both
    // defaulting True. Spelled 0/1 rather than parsed as a word, so a typo is an
    // integer refusal rather than a silent False.
    regenerate_video = ExtraInt(gen.extras, kLtx2RegenerateVideoExtra, 1) != 0;
    regenerate_audio = ExtraInt(gen.extras, kLtx2RegenerateAudioExtra, 1) != 0;
    if (im.pipeline_kind != "retake") {
      Fail("a retake request needs the 'retake' pipeline recipe, and this engine was loaded "
           "with pipeline_kind '" + im.pipeline_kind +
           "'. Upstream's retake runs ONE diffusion stage at the source clip's own resolution "
           "(retake.py:313-324, :317-318), and the distilled two-stage recipe renders its "
           "first stage at half — seeding that stage with a full-resolution source latent "
           "would put the clip into the wrong grid rather than fail");
    }
    if (!im.has_video_encoder) {
      Fail("this is a retake request and the video VAE checkpoint loaded here carries no "
           "encoder weights, so the source clip cannot become a latent at all "
           "(utils/helpers.py:229). Load a `video_vae_path` whose bag matches "
           "VAE_ENCODER_COMFY_KEYS_FILTER (video_vae/model_configurator.py:267-276)");
    }
    // The CLI-stage geometry validation, at the layer upstream runs it from:
    // BEFORE the pipeline is constructed (retake.py:344-353), so a clip off the
    // causal temporal grid or off the 32-pixel spatial grid costs no model work.
    retake_source = Ltx2ProbeFrameDirectory(gen.ref_video_dir);
    Ltx2RetakeAssertSourceGeometry(retake_source.frames, retake_source.height,
                                   retake_source.width, Ltx2ScaleFactors{}.time);
    // Upstream's retake CLI has no --height/--width/--num-frames at all
    // (`video_editing_arg_parser`, utils/args.py:848-877 omits them and says
    // "resolution comes from input video"). Preferring one of two geometries
    // silently is what this refuses.
    if (gen.height > 0 || gen.width > 0 || gen.num_frames > 1 || gen.duration_seconds > 0.0) {
      Fail("a retake request takes its geometry from the SOURCE clip and cannot also carry an "
           "explicit width, height, frame count or duration. Upstream's retake parser omits "
           "all of them for this reason (utils/args.py:848-877: \"no height/width/num-frames; "
           "resolution comes from input video\"), and the clip here is " +
           std::to_string(retake_source.width) + "x" + std::to_string(retake_source.height) +
           " at " + std::to_string(retake_source.frames) + " frames");
    }
  } else {
    for (const char* dependent : {kLtx2RetakeFrameRateExtra, kLtx2RegenerateVideoExtra,
                                  kLtx2RegenerateAudioExtra}) {
      if (!VideoExtra(gen.extras, dependent).empty()) {
        Fail("the '" + std::string(dependent) +
             "' extra was supplied without '" + std::string(kLtx2RetakeStartTimeExtra) +
             "' and '" + kLtx2RetakeEndTimeExtra +
             "', so there is no retake for it to configure. Refused rather than ignored");
      }
    }
  }

  if (!gen.prompt.empty() && !im.has_encoder) {
    Fail(
        "a prompt was supplied but no text tower is loaded, so it cannot condition this "
        "render. Rendering the prompt-embeds conditioning INSTEAD would silently ignore the "
        "request's own prompt. Load with encoder_path to condition on the prompt.");
  }
  if (gen.prompt.empty() && im.video_prompt_embeds.empty()) {
    if (im.has_encoder) {
      Fail(
          "this request carries no prompt, and no prompt-embeds conditioning was loaded "
          "either. A text tower IS loaded, so supply the prompt: rendering unconditioned "
          "would return a clip that looks like the tower not working.");
    }
    Fail("no conditioning is loaded; supply prompt_embeds_path and the '" +
         std::string(kLtx2AudioPromptEmbedsExtra) + "' extra");
  }

  // ── conditioning (phase L13) ──────────────────────────────────────────────
  //
  // THE PROMPT PATH, END TO END, and every step is a brick that already has a
  // golden — this composes them and adds no numerics:
  //
  //   prompt -> `Ltx2EncodePromptToConditioning`   (L10: tokenize on the embedded
  //             tokenizer, run the tower, aggregate all 49 hidden states, project
  //             them twice, right-pad-order the result)
  //          -> `RunConnector`                     (L9c: the checkpoint's OWN
  //             `*_embeddings_connector` weights, under its own `connector_*`
  //             configuration)
  //          -> `Ltx2ModalityInput::context`
  //
  // TWO THINGS A READER SHOULD BE ABLE TO CHECK RATHER THAN TAKE ON TRUST.
  //
  // (1) THE RIGHT-PAD SORT IS APPLIED TWICE AND THE SECOND IS THE IDENTITY.
  //     `Ltx2TextEncoderConditioning` and `Ltx2ConnectorCreateEmbeddings` are two
  //     ports of overlapping halves of `embeddings_processor.py:70-117`, and both
  //     carry the sort. Composing them therefore sorts an already-sorted stream.
  //     That is correct — a stable descending argsort of a 0/1 key is idempotent,
  //     which the first port's own header states — but "correct because it is
  //     idempotent" is a claim, so the loop below ASSERTS the precondition that
  //     makes it one: the additive mask leaving the encoder is non-increasing,
  //     i.e. every kept position precedes every padded one. If that ever stops
  //     holding the composition is refused rather than quietly re-ordering the
  //     caption against the mask.
  //
  //     THAT ASSERTION IS ITSELF UNGATED, and it is named here rather than left
  //     to read as a positive. No test reaches it: every mask that arrives comes
  //     from `Ltx2ComputeRightPadOrder`, which produces a non-increasing mask by
  //     construction, so the only way to fire it through the public API is a
  //     defect INSIDE that function. Deleting the loop moves no test. That is
  //     NOT a claim that it is unreachable — no probe was built that fails to
  //     reach it, and this campaign's own rule is that a mutation which moves
  //     nothing is not evidence of unreachability. It is a guard on an internal
  //     invariant, carried because the invariant is what makes the double sort
  //     an identity, and recorded as ungated.
  //
  // (2) THE TOWER RUNS ON A CPU QUEUE EVEN ON THE DEVICE ARM, and that is a
  //     stated limit, not an oversight. Everything in `ltx2_text_encoder.h` is
  //     f32 by declaration — `Ltx2TextFeatureExtractorForward` REFUSES any other
  //     compute dtype by name — so the text path has no device arm to run on.
  //     The DiT still runs where `device` says. What this costs is a host-side
  //     12B forward over the prompt's own valid tokens once per request, against
  //     a denoise loop of many 21B forwards; it is recorded as owed rather than
  //     hidden, and it is the reason `im.queue` is not passed here.
  setup_phase.Close();

  std::vector<float> prompt_video, prompt_audio;
  const float* video_context = im.video_prompt_embeds.data();
  const float* audio_context = im.audio_prompt_embeds.data();
  int64_t context_tokens = im.prompt_tokens;
  im.trace = Ltx2ConditioningTrace{};

  if (!gen.prompt.empty()) {
    // W0: the phase #1269 and W4 are about. Split into the TOWER and the
    // CONNECTOR because they are different work on different weights, and the
    // spike's 39-100% bound could not tell them apart.
    const phase::Scope conditioning_span("generate.conditioning", /*span=*/true);
    vt::Queue text_queue{vt::Device{vt::DeviceType::kCPU, 0}, nullptr};
    phase::Scope tower_phase("conditioning.tower");
    const Ltx2PromptConditioning encoded = Ltx2EncodePromptToConditioning(
        *im.tower, *im.tokenizer, im.gemma_ids, im.caption_projections, im.feature_cfg,
        gen.prompt, text_queue);
    tower_phase.Close();
    prompt_video = encoded.conditioning.video;
    prompt_audio = encoded.conditioning.audio;
    context_tokens = encoded.seq;

    if (im.has_connector) {
      const std::vector<float>& mask = encoded.conditioning.additive_mask;
      if (static_cast<int64_t>(mask.size()) != context_tokens) {
        Fail("the text encoder returned a " + std::to_string(mask.size()) +
             "-entry additive mask for " + std::to_string(context_tokens) + " rows");
      }
      for (size_t s = 1; s < mask.size(); ++s) {
        if (mask[s] <= mask[s - 1]) continue;
        Fail(
            "the text encoder's additive mask is not right-padded at row " +
            std::to_string(s) +
            ", so the connector's own right-pad sort would REORDER it rather than "
            "leave it alone. The two ports of embeddings_processor.py:23-43 compose "
            "only because a stable argsort of a 0/1 key is idempotent, and that holds "
            "only on an already right-padded input. Refusing rather than sorting a "
            "caption stream against a mask it no longer matches.");
      }
      phase::Scope connector_phase("conditioning.connector");
      const Ltx2ConnectorEmbeddings through = RunConnector(
          SafetensorsFile::Open(im.params.dit_path), im.video_connector_cfg,
          im.audio_connector_cfg, prompt_video, prompt_audio, mask, context_tokens);
      connector_phase.Close();
      prompt_video = through.video;
      prompt_audio = through.audio;
    }
    video_context = prompt_video.data();
    audio_context = prompt_audio.data();
    im.trace.from_prompt = true;
    im.trace.prompt = gen.prompt;
  }

  {
    const int64_t vw = im.dit.params.cross_attention_dim;
    const int64_t aw = im.dit.params.audio_cross_attention_dim;
    const std::vector<float>& v = im.trace.from_prompt ? prompt_video : im.video_prompt_embeds;
    const std::vector<float>& a = im.trace.from_prompt ? prompt_audio : im.audio_prompt_embeds;
    if (static_cast<int64_t>(v.size()) != context_tokens * vw ||
        static_cast<int64_t>(a.size()) != context_tokens * aw) {
      Fail("the conditioning is " + std::to_string(v.size()) + " / " +
           std::to_string(a.size()) + " floats for " + std::to_string(context_tokens) +
           " rows at widths " + std::to_string(vw) + " / " + std::to_string(aw));
    }
    im.trace.tokens = context_tokens;
    im.trace.video_width = vw;
    im.trace.audio_width = aw;
    im.trace.video_digest = DigestF32(v);
    im.trace.audio_digest = DigestF32(a);
    im.trace.video_absmax = AbsMax(v);
    im.trace.audio_absmax = AbsMax(a);
  }

  // ── TEXT-TO-AUDIO: the render that has no picture (#1005) ─────────────────
  //
  // `T2AOneStagePipeline.__call__` (t2a_one_stage.py:109-172), in its own
  // translation unit (ltx2_t2a.h) mirroring upstream's own file.
  //
  // THE BRANCH SITS HERE, after the conditioning and before ANY video geometry.
  // After, because T2A encodes a prompt exactly as every other pipeline does
  // (`:127-135`) and duplicating that chain would give the audio-only arm its own
  // copy of the connector composition. Before, because everything below this
  // point — the resolution guard, the canvas, the phase loop, the decode — is
  // about a video stream this pipeline does not have, and a `t2a` request that
  // fell through would be refused by a message about latent grids.
  if (im.recipe.audio_only) {
    VideoResult audio_only = GenerateAudioOnly(im, gen, audio_context, context_tokens);
    generate_span.Close();
    WritePhaseLog(gen.output_dir, kLtx2VideoFamily, phase_device, &audio_only.phase_log_path);
    return audio_only;
  }

  // W0: the five leaves below CHAIN — each one closes where the next opens —
  // so the driver's linear prologue is named rather than left as residue.
  phase::Scope prep_image("generate.image_cond");
  // ── conditioning on pixels (row LTX25-IMAGE-COND, issue #644) ─────────────
  //
  // Upstream this is `ImageConditioner` (ltx-pipelines/utils/blocks.py:936-993,
  // called at distilled.py:212) feeding `combined_image_conditionings`
  // (utils/helpers.py:272-308). ONE of its four arms is served here, and the
  // other three are refused BY NAME rather than dropped — a keyframe that is
  // silently ignored renders an unconditioned clip that looks like the feature
  // not working.
  //
  // THESE MESSAGES ARE WRITTEN TO BE RE-CHECKABLE, and the count is still SIX
  // refusals in this campaign whose stated reason turned out to be false or
  // stale. Two of the six stood right here. The first said no encoder weights
  // could be materialized — true when written, and what this row fixed. The
  // second replaced it and blamed `keyframes_abs_pos_embedding`, which was
  // verifiably NOT the blocker at the pin (see the last-frame message below for
  // the three anchors that refute it), and a test had been written to assert
  // that wrong reason by name.
  //
  // IT STAYS AT SIX, AND TWO MORE NEARLY JOINED IT ON THE SAME DAY. `c7cb59fbb`
  // (row LTX25-TOKEN-APPEND, #930) built the append seam and falsified the
  // stated cause of TWO refusals that were open in review when it landed: the
  // generated-keyframe-slot one (#920) and the reference one below (#923).
  // Neither reached `main`, so neither is counted; both are recorded, because
  // the MECHANISM is the point and it was identical. Each had been rewritten
  // onto token-append days earlier, and each was gated by assertions on
  // UPSTREAM symbol names and on literals the message declared about itself —
  // and no change to THIS engine can move either kind. Both repairs have the
  // same shape: assert a property of this tree, then constrain the message
  // against it. The reference case below measures that the phase loop grows and
  // trims before it reads one character of the refusal; the slot case re-derives
  // the message's own `DECLARED HERE` / `ABSENT HERE` lists out of
  // `ltx2_conditioning.h`.
  //
  // So: name the exact symbol or upstream `file:line` that would have to change
  // for the refusal to become false, never a category — and where a plausible
  // reason has already been ruled OUT, say so and cite what ruled it out, so the
  // next reader re-checks the claim instead of re-deriving the refutation. Local
  // anchors are SYMBOLS, not line numbers in this file: same-file line numbers
  // drift on every edit, which is how the previous message's citation went stale.
  // TWO ARMS OF ONE UPSTREAM LOOP. `combined_image_conditionings`
  // (ltx-pipelines/utils/helpers.py:272-308) iterates one `images` list and
  // branches per item: `frame_idx == 0` takes `VideoConditionByLatentIndex`,
  // anything else takes `VideoConditionByKeyframeIndex`. So the CRF, the
  // strength polarity, the preprocess and the encode are shared by construction
  // upstream, and they are shared here for the same reason rather than
  // duplicated per arm.
  //
  // Row LTX25-TOKEN-APPEND (#930) opened the second branch. The refusal that
  // stood here named the token-append machinery, and that refusal was accurate:
  // a keyframe APPENDS (keyframe_cond.py:79-82) where an image at latent frame 0
  // REPLACES (latent_cond.py:38-39), and this loop had no way to grow the
  // sequence and trim it back. `Ltx2ExtendKeyframesMask` and
  // `Ltx2ClearConditioning` are the two halves it was missing.
  const bool wants_first_frame = !gen.first_frame_path.empty() || !gen.first_frame_ppm.empty();
  const bool wants_last_frame = !gen.last_frame_path.empty();
  const bool wants_image = wants_first_frame || wants_last_frame;
  // `!wants_retake` IS LOAD-BEARING, and it narrows this refusal rather than
  // weakening it. Retake and IC-LoRA reference conditioning both arrive as
  // `ref_video_dir`, and they consume it in completely different ways: retake
  // encodes the clip at its OWN resolution and seeds the video stream's initial
  // latent with it (retake.py:238-247, :273), while the reference item is
  // downscaled by the adapter's factor, temporally subsampled, and APPENDED as
  // extra tokens to a stage-1-only adapter (iclora_utils.py:112-117, :87-89,
  // :144-148). Serving the first says nothing about the second, so the second
  // stays refused and #975 stays open.
  if (!wants_retake && (!gen.ref_image_paths.empty() || !gen.ref_video_dir.empty())) {
    // TWO CAUSES REMAIN, AND NEITHER IS ONE THIS MESSAGE HAS EVER GIVEN. The
    // message names both, and then names the three ruled-out reasons with what
    // ruled each one out, because a reader who arrives here in a month should
    // re-check the claim rather than re-derive the refutation for a third time.
    //
    // 1. THE REFERENCE CLIP HAS NO PIXEL PATH. Upstream resolves the reference
    //    at `height // scale` by `width // scale` (iclora_utils.py:116-117),
    //    refuses a target either axis of which the factor does not divide
    //    (:112-115), keeps frame 0 and then every Nth frame (`temporal_subsample`,
    //    :87-89, called at :144), and encodes the whole clip (:145-148). This
    //    engine's only pixel-to-latent route for a REFERENCE item is
    //    `Ltx2LoadImageAndPreprocess` followed by `Ltx2ConvVideoEncode` at
    //    `frame_count = 1` and the phase's OWN height and width, and it refuses
    //    an encode that returns more than one latent frame.
    //
    //    THIS USED TO SAY "nothing anywhere reads `ref_video_dir`", and that was
    //    false about the tree even when it was written (#987): MiniMax-H3 has
    //    always consumed the directory in full — `ReadReferenceClipChw`,
    //    `minimax_h3_video.cpp:135`, called at `:650`. Since row LTX25-RETAKE
    //    (#924) the LTX-2.5 side reads it too, through
    //    `Ltx2ReadFrameDirectory`. So the missing piece is NOT a reader. It is
    //    the reference item's own geometry: the downscale-factor resize and the
    //    temporal subsample, neither of which retake performs and neither of
    //    which any reader supplies.
    //
    //    THE SECOND REASON THIS MESSAGE GAVE IS NOW FALSE, and it is recorded
    //    here rather than deleted because it is the third reason in this block
    //    to come true and a reader needs to know which. It said: "the reference
    //    item belongs to stage 1 and stage 2 must run unfused —
    //    `ICLoraPipeline` gives stage 1 `loras=tuple(loras)` (ic_lora.py:108)
    //    and stage 2 `loras=()` (:119), and this engine holds ONE `Ltx2Dit`,
    //    fused at load, that every phase of the recipe runs. Serving the arm
    //    needs a second unfused DiT or a phase-scoped adapter."
    //
    //    Row LTX25-PHASE-LORA (#1118) landed the phase-scoped adapter.
    //    `Ltx2PhaseRecipe::loras` (ltx2_pipeline.h) carries upstream's per-stage
    //    set and the phase loop in this file honours it through
    //    `Ltx2RebindDitLoras`, which re-materializes only the tensors an adapter
    //    targets — so a two-phase recipe CAN now give stage 1 the adapter and
    //    stage 2 none, which is exactly `ic_lora.py:108` against `:119`, and it
    //    does so without a second resident weight set. `A2VidTwoStageRecipe` is
    //    the executable proof it exists: it gives stage 1 `kNoAdapters` and the
    //    gate "the distilled adapter rides stage 2 ALONE" renders both states
    //    through this ABI and compares the pixels.
    //
    //    What that leaves is reason 1 ALONE, and reason 1 is unrelated to
    //    weights: it is the reference clip's own geometry. The conditioning
    //    split is also still upstream's — stage 1 takes `_create_conditionings`,
    //    which appends the reference item (:269-278, :377-402), and stage 2
    //    takes plain `combined_image_conditionings` with no reference item
    //    (:314-321) — but that is a conditioning question, not a fused-weight
    //    one, and serving the arm on one phase only is upstream's `skip_stage_2`
    //    (:302-308), a different request.
    std::string factors = "no adapter was supplied, so none were read";
    if (im.dit.lora_fused_tensors > 0) {
      factors = "the supplied adapter declares downscale=" +
                std::to_string(im.dit.lora_reference.downscale) +
                " temporal=" + std::to_string(im.dit.lora_reference.temporal) +
                ", fused into " + std::to_string(im.dit.lora_fused_tensors) + " tensors";
    }
    Fail(
        "reference-image / reference-video conditioning is not served. TWO things are "
        "missing. FIRST, the reference CLIP has no pixel path: upstream reads it at "
        "`height // reference_downscale_factor` by `width // reference_downscale_factor` "
        "(iclora_utils.py:116-117), refuses a target the factor does not divide (:112-115), "
        "keeps frame 0 and then every Nth frame (`temporal_subsample`, :87-89, called at "
        ":144) and encodes the whole clip (:145-148), while this engine's only "
        "pixel-to-latent route for a REFERENCE item encodes exactly ONE frame at the phase's "
        "own resolution. WHAT IS *NOT* THE REASON here: the READER. This message used to say "
        "\"nothing reads `ref_video_dir` at all\", which was false about the tree when it was "
        "written (#987) — MiniMax-H3 consumes the directory in full at "
        "`minimax_h3_video.cpp:650` — and is doubly false now that row LTX25-RETAKE (#924) "
        "reads it on this side through `Ltx2ReadFrameDirectory`. What is missing is the "
        "reference item's own geometry, the downscale resize and the temporal subsample, "
        "which no reader supplies. SECOND, the reference item is a STAGE-1 item and stage 2 "
        "takes `combined_image_conditionings` with no reference item at all: `ICLoraPipeline` "
        "gives stage 1 the reference conditioning (ic_lora.py:269-278) and stage 2 none "
        "(:314-321), and this phase loop appends the same conditioning set to every phase. "
        "That is a CONDITIONING gap and not a weights one. WHAT IS *NOT* THE REASON, because "
        "this refusal has now given THREE reasons that later became false: (a) the IC-LoRA "
        "METADATA. Row LTX25-IC-LORA (#923) "
        "closed that; supply `lora_path` and the factors are read at load "
        "(iclora_utils.py:30-49) — right now, " + factors +
        ". (b) the TOKEN-APPEND machinery. This message blamed it on 2026-08-15 and row "
        "LTX25-TOKEN-APPEND (#930) landed it in `c7cb59fbb` the next day: the phase loop "
        "now binds a `target_tokens` local, grows `video.tokens` past it on an appending "
        "item, carries the grown count through denoise, and trims back through "
        "`Ltx2ClearConditioning` (ltx_core/tools.py:88-117) before unpatchify. The "
        "last-frame keyframe arm is SERVED on exactly that machinery, which is the "
        "executable proof it exists. (c) `Ltx2LatentState` carrying no attention-mask "
        "field. On the DEFAULT arm upstream builds no mask: at "
        "`conditioning_attention_strength >= 1.0` with no latent mask `attn_mask` is None "
        "(iclora_utils.py:159-160) and `ConditioningItemAttentionStrengthWrapper` is "
        "applied only `if attn_mask is not None` (:168-169). The sub-1.0 arm is owed by "
        "#932, and it is not what blocks this one. (d) the FUSED-AT-LOAD adapter. This "
        "message said until 2026-08-17 that stage 2 must run with no adapter while \"this "
        "engine holds one DiT, fused at load, that every phase runs\", and row "
        "LTX25-PHASE-LORA (#1118) closed it: `Ltx2PhaseRecipe::loras` carries upstream's "
        "per-stage set and the phase loop rebinds the DiT through `Ltx2RebindDitLoras`, so "
        "`loras=tuple(loras)` on stage 1 against `loras=()` on stage 2 (ic_lora.py:108, "
        ":119) is now expressible with no second weight set. `a2vid_two_stage`'s stage 1 "
        "runs `kNoAdapters` on exactly that machinery, which is the executable proof it "
        "exists. Use first_frame_ppm / first_frame_path "
        "for image-to-video, and last_frame_path for a closing keyframe.");
  }
  if (!gen.ref_audio_path.empty() || !gen.ref_audio_wav.empty()) {
    Fail(
        "reference-AUDIO conditioning is not served. `Ltx2ConditionAudioByReference` is ported "
        "and gated (conditioning/types/reference_audio_cond.py:34-65), and what it needs is an "
        "encoded waveform: `encode_audio` through the audio VAE's ENCODER "
        "(ltx-pipelines/utils/helpers.py:264-269). What is missing is the CONDITIONING ITEM's "
        "delivery: `Ltx2ConditionAudioByReference` appends reference tokens to the audio "
        "stream, and nothing in this phase loop constructs one from a request. WHAT IS *NOT* "
        "THE REASON: the audio VAE ENCODER. This message used to say \"there is no "
        "AUDIO_VAE_ENCODER key filter\", and row LTX25-A2V-AUDIO-INPUT (#922) landed one in "
        "`c2019b0e3` — `Ltx2AudioVaeEncoderKeyRules()` is loaded above and "
        "`Ltx2EncodeAudioToLatent` is called on the audio-to-video path in this same "
        "function, which is the executable proof it exists (#987). Recorded as owed.");
  }

  // The CRF, resolved the way `ImageConditioner.resolve_crf` resolves it
  // (blocks.py:966-983) over `detect_params` (utils/constants.py:166-179): from
  // the CHECKPOINT's own generation when the caller left it unset. For LTX-2.5
  // that is 18, and 18 is not ported — so the DEFAULT REFUSES and a caller has
  // to ask for 0 knowingly. Resolved and checked BEFORE any pixel is read, so an
  // unsupported request costs nothing and reports the same thing every time.
  int64_t image_crf = 0;
  double image_strength = 0.0;
  std::string image_bytes, last_frame_bytes;
  if (wants_image) {
    if (!im.has_video_encoder) {
      Fail(
          "an image conditioning was supplied but the video VAE checkpoint at '" +
          im.params.video_vae_path +
          "' carries no ENCODER half: no tensor in it is named `vae.encoder.*` or `encoder.*`, "
          "which is what `VAE_ENCODER_COMFY_KEYS_FILTER` matches "
          "(video_vae/model_configurator.py:267-276). A Comfy-split `vae/` file holding the "
          "decoder alone reads exactly like this. Supply the monolithic VAE checkpoint, or the "
          "encoder file, rather than rendering unconditioned.");
    }
    image_crf = ExtraInt(gen.extras, kLtx2ImageCrfExtra,
                         Ltx2ResolveDefaultImageCrf(Ltx2ParseModelVersion(im.model_version)));
    // Throws by name at any non-zero value, naming the unported codec round
    // trip and saying that 0 is the supported — and out-of-distribution — one.
    Ltx2PreprocessImageCrf(image_crf);

    // `ImageConditioningInput.strength` (utils/args.py:64). The seam's
    // `noise_aug` is documented as "keyframe pinning strength; <= 0 => 1.0"
    // (include/vllm.h), which is the same polarity: 1 pins, and the item turns
    // it into `denoise_mask = 1 - strength` (latent_cond.py:41).
    image_strength = gen.noise_aug > 0.0 ? gen.noise_aug : 1.0;
    if (image_strength > 1.0) {
      Fail("the image conditioning strength is " + std::to_string(image_strength) +
           "; upstream's denoise mask is `1 - strength` (latent_cond.py:41) and a strength "
           "above 1 makes it negative, which the noiser extrapolates PAST the clean latent "
           "rather than toward it (components/noisers.py:33)");
    }
    if (wants_first_frame) {
      image_bytes = gen.first_frame_ppm.empty()
                        ? ReadFileBytes("first_frame", gen.first_frame_path)
                        : gen.first_frame_ppm;
    }
    if (wants_last_frame) last_frame_bytes = ReadFileBytes("last_frame", gen.last_frame_path);
    im.trace.image_crf = image_crf;
    im.trace.image_strength = image_strength;
  }

  prep_image.Close();
  phase::Scope prep_geometry("generate.geometry");
  // ── geometry ──────────────────────────────────────────────────────────────
  const Ltx2PipelineRecipe& recipe = im.recipe;
  // A retake's geometry is the SOURCE clip's, not the recipe's and not the
  // request's: `output_shape = get_videostream_metadata(video_path, fps=fps)`
  // (retake.py:220) and every one of the four is passed straight to the stage
  // (:317-320). The request cannot carry a conflicting one — that is refused
  // above — so this is a substitution rather than a precedence rule.
  const double fps = wants_retake ? retake_fps : recipe.frame_rate;
  const int64_t height = wants_retake ? retake_source.height
                                      : (gen.height > 0 ? gen.height : recipe.height);
  const int64_t width = wants_retake ? retake_source.width
                                     : (gen.width > 0 ? gen.width : recipe.width);
  int64_t frames =
      wants_retake ? retake_source.frames : (gen.num_frames > 1 ? gen.num_frames : recipe.num_frames);
  if (gen.duration_seconds > 0.0) {
    // `resolve_num_frames` (utils/blocks.py) turns an AUTO duration into frames
    // through the DURATION HEAD. THE REASON THIS IS UNSERVED MOVED IN L13 and the
    // old one is recorded so a reader can re-check it: it used to be "the head
    // needs an encoded prompt this engine cannot produce", and since `has_encoder`
    // above the engine produces exactly that. What is missing now is the head
    // itself — `ltx2_duration_head.h` is ported and gated as a brick, but nothing
    // here constructs one. `duration_head_path` used to be ACCEPTED while no code
    // read it, so a caller who supplied a head silently landed on this line
    // instead; `CheckUnservedExtras` now refuses that key by name at load (#611,
    // .agents/specs/ltx25-retire-dead-arms.md §2). What remains owed is the head
    // itself. An explicit duration is exact arithmetic, so it is served; the AUTO
    // path is what is missing, and `num_frames` is how to avoid it.
    frames = static_cast<int64_t>(std::llround(gen.duration_seconds * fps));
  }
  if (frames < 1) Fail("num_frames resolved to " + std::to_string(frames));

  const Ltx2ScaleFactors factors;  // VIDEO_SCALE_FACTORS (types.py:70) — the conv
                                   // arm's fixed (8, 32, 32), not derived
                                   // (utils/helpers.py:66-72)

  // ── DFR: the canvas, and the slot grid it decides (#986) ──────────────────
  //
  // Declared here and RESOLVED below `Ltx2AssertResolution`, because upstream's
  // order is `assert_resolution` at dfr_pipeline.py:291 and `resolve_canvas` at
  // :314, in that order. A request that is wrong on BOTH axes must hear about
  // the resolution first, exactly as it would upstream; otherwise the two ports
  // give different answers to the same bad request and only one of them matches
  // the reference.
  //
  // `requested_frames` is captured BEFORE the pad, because it is the contract.
  const bool is_dfr = im.pipeline_kind == "dfr";
  const int64_t requested_frames = frames;
  std::vector<int64_t> slot_positions;

  // `assert_resolution` (utils/helpers.py:540-551), at the position upstream
  // calls it from: the top of `__call__`, before any work is paid for. The
  // divisor is DERIVED — the VAE spatial factor times the worst downscale this
  // recipe's phases apply — which is upstream's own 64 for a two-stage recipe and
  // 32 for a one-stage one, reached by upstream's reasoning rather than restated
  // as two literals.
  //
  // FRAMES ARE DELIBERATELY NOT CHECKED HERE, and the asymmetry is upstream's.
  // `resolve_num_frames` (utils/blocks.py:908-928) returns an explicit count
  // verbatim (utils/blocks.py:920-921) and `VideoLatentShape.from_pixel_shape`
  // (ltx_core/types.py:113)
  // then floors it exactly as the `vshape.frames = (frames - 1) / factors.time + 1`
  // line in the phase loop below does. Adding a refusal here would be a divergence
  // from the reference, not a mirror of it — so `docs/USAGE.md` carries the
  // rounding as documented behaviour instead (#919).
  //
  // `snap_frames_to_grid` (utils/helpers.py:554-562) does NOT contradict that,
  // and the reason is not the one it is easy to give. It has three callers, not
  // one: utils/helpers.py:581 inside `seconds_to_clamped_num_frames`, which is the
  // auto-duration path, and dubit.py:215 and :396, the second of which is inside
  // `DubitPipeline.__call__` three lines after its own `assert_resolution`. So
  // "only the auto-duration path snaps" is false. What holds is sharper:
  // `DubitPipeline.__call__` takes NO `num_frames` at all (dubit.py:194-210) and
  // snaps a count it read from the reference video's container metadata. Counted
  // at the pin, it is the only pipeline `__call__` that snaps, and the only one
  // with no `num_frames` parameter — every `__call__` that does take one leaves it
  // unsnapped. An explicit frame count is floored upstream and here, and validated
  // in neither.
  // ONE divisor for both axes, as upstream has (`divisor = 64 if is_two_stage
  // else 32`). That is a mirror and not a simplification: upstream's
  // VIDEO_SCALE_FACTORS is (8, 32, 32), so its single spatial divisor already
  // covers both axes. The equality is asserted rather than assumed, because a VAE
  // whose axes differed would otherwise have its width checked against the height
  // factor and no test would see it.
  if (factors.height != factors.width) {
    Fail("the VAE's spatial scale factors differ (" + std::to_string(factors.height) +
         " high, " + std::to_string(factors.width) +
         " wide), so one resolution divisor cannot cover both axes the way "
         "`assert_resolution` (utils/helpers.py:540-551) does");
  }
  Ltx2AssertResolution(height, width, factors.height * recipe.max_spatial_downscale());

  // ── DFR: resolve the canvas (#986) ────────────────────────────────────────
  //
  // `resolve_canvas` (dfr_layout.py:60-81) at `dfr_pipeline.py:314`, after
  // `assert_resolution` and before the first stage. It PADS the request up to a
  // whole number of keyframe segments, so the canvas this engine denoises is
  // generally LONGER than what the caller asked for, and `:534` trims back at
  // the end. Both halves are needed: dropping the pad would put the terminal
  // keyframe off a segment boundary, and dropping the trim would hand the caller
  // a clip longer than the one they requested.
  if (is_dfr) {
    // Upstream's admission gate, at upstream's position: the pipeline preamble,
    // before any weight is built (`assert_generated_keyframes_supported`,
    // utils/blocks.py:405-419, called from dfr_pipeline.py:315).
    AssertGeneratedKeyframesSupported(im.dit.params.use_keyframes_abs_pos_embedding,
                                      im.params.dit_path);
    const Ltx2DfrCanvas canvas = Ltx2DfrResolveCanvas(frames, factors.time);
    frames = canvas.num_frames;
    slot_positions = canvas.positions;
    im.trace.canvas_frames = canvas.num_frames;
    im.trace.canvas_segment = canvas.segment;
  } else {
    // Every other pipeline takes the slot COUNT and spaces the positions evenly
    // (`resolve_generated_keyframes`, utils/helpers.py:394-411). Resolved here
    // rather than in the extras check because the count needs `frames`, which is
    // not known at that point.
    slot_positions = EvenlySpacedKeyframePositions(
        ExtraInt(gen.extras, kLtx2GeneratedKeyframesExtra, 0), frames);
    if (!slot_positions.empty()) {
      AssertGeneratedKeyframesSupported(im.dit.params.use_keyframes_abs_pos_embedding,
                                        im.params.dit_path);
    }
  }

  // `AudioLatentShape.from_video_pixel_shape` (types.py:184-200) and
  // `VideoLatentShape.from_pixel_shape` (:108-123) defaults. Asserted against the
  // DiT rather than assumed: the audio latent's channels x mel_bins IS the audio
  // stream's input width, and a mismatch would reinterpret the spectrogram.
  constexpr int64_t kAudioLatentChannels = 8;
  constexpr int64_t kAudioLatentMelBins = 16;
  if (kAudioLatentChannels * kAudioLatentMelBins != im.dit.params.audio_in_channels) {
    Fail("the audio latent is " + std::to_string(kAudioLatentChannels) + " x " +
         std::to_string(kAudioLatentMelBins) + " = " +
         std::to_string(kAudioLatentChannels * kAudioLatentMelBins) +
         " wide (types.py:184-200) but the DiT's audio stream takes " +
         std::to_string(im.dit.params.audio_in_channels));
  }

  const int64_t last_phase =
      im.max_phase >= 0 ? im.max_phase : static_cast<int64_t>(recipe.phases.size()) - 1;

  // One generator for the state noise (distilled.py:214-215) — see
  // SplitMixGaussian for what this is and is not.
  const uint64_t seed = gen.has_seed ? gen.seed : static_cast<uint64_t>(recipe.num_frames);
  SplitMixGaussian state_noise(seed);

  std::vector<float> video_latent_volume;  // [C, F, H, W], unpatchified
  int64_t video_lc = 0, video_lf = 0, video_lh = 0, video_lw = 0;
  // The generated keyframe slots as the LAST phase that ran produced them, and
  // the layout that locates them. Carried ACROSS phases: stage 2 seeds its slots
  // with stage 1's, spatially upsampled (dfr_pipeline.py:348, :364).
  Ltx2LatentVolume slot_keyframes;
  Ltx2GeneratedKeyframeLayout slot_layout;
  std::vector<float> audio_latent_volume;  // [C, F, M], unpatchified
  int64_t audio_lc = 0, audio_lf = 0, audio_lm = 0;

  prep_geometry.Close();
  phase::Scope prep_audio("generate.audio_input");
  // ── AUDIO-TO-VIDEO: the driving waveform (#922) ────────────────────────────
  //
  // `A2VidPipelineTwoStage.__call__` lines 196-202: decode the file, encode it
  // through the audio VAE, truncate it to the clip's own duration. The result
  // then rides FROZEN through every phase (`ModalitySpec(frozen=True,
  // noise_scale=0.0)`, :251-256 and :291-296), which is what makes this
  // audio-to-video rather than joint generation: the soundtrack is an input, and
  // the video is denoised around it.
  //
  // Done ONCE, before the phase loop, and deliberately not per phase: upstream
  // encodes once at :200 and hands the SAME tensor to both stages. Re-encoding
  // per phase would be pure waste on a path where the latent cannot change.
  std::vector<float> a2v_audio_volume;
  // The take as DECODED, kept for the output. Upstream returns the caller's own
  // waveform rather than the VAE's reconstruction of it, in as many words:
  // "Return the original input audio instead of VAE-decoded audio to preserve
  // fidelity" (a2vid_two_stage.py:301-303). Round-tripping a file the caller
  // already has through an encoder and a vocoder can only lose to it.
  Ltx2DecodedAudio a2v_source;
  const std::string a2v_audio_path = VideoExtra(gen.extras, kLtx2AudioPathExtra);
  // REQUIRED on a recipe that says so (#1117). `--audio-path` is `required=True`
  // (a2vid_two_stage.py:312-317), and the pipeline's whole shape is "denoise
  // video AROUND this take": both stages freeze the audio stream on it
  // (`:251-256`, `:291-296`) and the soundtrack handed back is the caller's own
  // file (`:301-303`).
  //
  // Checked HERE and not at load, because `pipeline_kind` is a LOAD extra and
  // `audio_path` is a per-generation one, so the question is only decidable once
  // a request exists. Keyed on the recipe flag rather than on the kind string,
  // for the reason `audio_only` gives in the header.
  //
  // WITHOUT THE TAKE THE RENDER STILL FINISHES. The audio stream is generated
  // rather than supplied, which is ordinary joint generation, and the result is
  // a clip of the right size with the right frame count and the right sample
  // rate — indistinguishable from audio-to-video that ignored its input.
  if (im.recipe.requires_audio_input && a2v_audio_path.empty()) {
    Fail("the '" + im.pipeline_kind + "' pipeline is driven BY a waveform and no '" +
         std::string(kLtx2AudioPathExtra) +
         "' extra was supplied. Upstream's `--audio-path` is `required=True` "
         "(ltx-pipelines a2vid_two_stage.py:312-317) and both of its stages freeze the audio "
         "stream on the encoded take (`:251-256`, `:291-296`). Refused rather than rendered: "
         "without it the soundtrack is GENERATED, and a generated one is a finished clip at the "
         "right size, frame count and sample rate with nothing to show that the input was "
         "ignored. Supply the take, or load with a `pipeline_kind` that generates audio.");
  }
  if (!a2v_audio_path.empty()) {
    if (!im.has_audio_encoder) {
      Fail("'" + std::string(kLtx2AudioPathExtra) + "' names '" + a2v_audio_path +
           "', but the audio VAE checkpoint loaded here carries no `audio_vae.encoder.` "
           "tensors, so there is nothing to turn that waveform into latents with. "
           "`Ltx2AudioEncoderForward` and its mel front-end are ported and gated "
           "(audio_vae.py:190-246, ops.py:8-55); what this checkpoint is missing is the "
           "WEIGHTS. Supply an audio VAE that carries the encoder half. Refused rather than "
           "rendering the clip unconditioned, which would look like the feature working");
    }
    // `AudioLatentShape.from_duration` (types.py:164-181), the same expression
    // the phase loop uses for `ashape.frames` below. Computed here so the
    // truncation target and the stream's token count cannot drift apart.
    const Ltx2AudioPatchifierParams ap;
    const double latents_per_second = static_cast<double>(ap.sample_rate) /
                                      static_cast<double>(ap.hop_length) /
                                      static_cast<double>(ap.audio_latent_downsample_factor);
    const int64_t target_frames = static_cast<int64_t>(
        std::llround(static_cast<double>(frames) / fps * latents_per_second));

    // `--audio-max-duration` defaults to the CLIP's duration, and that default
    // is applied by upstream's CLI (a2vid_two_stage.py:369-371), not by the
    // pipeline, whose own default is None (:157). Mirrored at the same layer.
    const double start_time =
        ExtraDouble(gen.extras, kLtx2AudioStartTimeExtra, 0.0);
    const double max_duration =
        ExtraDouble(gen.extras, kLtx2AudioMaxDurationExtra, static_cast<double>(frames) / fps);
    if (start_time < 0.0) {
      Fail("'" + std::string(kLtx2AudioStartTimeExtra) + "' is " + std::to_string(start_time) +
           "; it is a position in seconds and cannot be negative");
    }
    if (max_duration <= 0.0) {
      Fail("'" + std::string(kLtx2AudioMaxDurationExtra) + "' is " + std::to_string(max_duration) +
           "; it is a duration in seconds and must be positive");
    }

    a2v_source = Ltx2DecodeAudioWav(
        ReadFileBytes(kLtx2AudioPathExtra, a2v_audio_path), a2v_audio_path,
        im.audio_encoder_cfg.encoder.in_channels,
        im.audio_encoder_cfg.processor.target_sample_rate, start_time, max_duration);

    const Ltx2AudioSpectrogram encoded = Ltx2EncodeAudioToLatent(
        im.audio_encoder_cfg.encoder, im.audio_encoder_cfg.processor, im.audio_encoder_weights,
        a2v_source, target_frames);

    // The DiT's audio stream is `channels x mel_bins` wide and the engine
    // already refuses a checkpoint whose product disagrees. Check the encoder's
    // OWN output against the same two numbers rather than only the product: a
    // (16, 8) latent has the same width as an (8, 16) one and unpatchifies into
    // a different tensor.
    if (encoded.channels != kAudioLatentChannels || encoded.mel_bins != kAudioLatentMelBins) {
      Fail("the audio VAE encoder produced a " + std::to_string(encoded.channels) + " x " +
           std::to_string(encoded.mel_bins) +
           " latent and this DiT's audio stream takes " + std::to_string(kAudioLatentChannels) +
           " x " + std::to_string(kAudioLatentMelBins) + " (types.py:184-200)");
    }
    a2v_audio_volume = encoded.data;
  }

  prep_audio.Close();
  phase::Scope prep_retake("generate.retake");
  // ── RETAKE: the source clip becomes the initial video latent (#924) ───────
  //
  // `RetakePipeline.__call__` lines 238-247: read the clip and encode it through
  // the video VAE, then `_conform_latent_length` it to the frame count the
  // target grid needs (utils/helpers.py:230-233). Done ONCE, before the loop,
  // because upstream encodes once and the latent cannot change.
  //
  // THE AUDIO HALF DOES NOT EXIST ON THIS ARM, and that is upstream's rather
  // than an omission: `audio_latent_from_file` returns None for a frame folder
  // before it opens anything (utils/helpers.py:261-262), so
  // `initial_audio_latent` is None, and BOTH of retake's audio predicates are
  // conjunctions with `initial_audio_latent is not None` (retake.py:279, :282).
  // `Ltx2RetakePlanModalities` is where that is expressed, and the soundtrack is
  // generated fresh.
  std::vector<float> retake_video_volume;  // [C, F, H, W], unpatchified
  Ltx2RetakePlan retake_plan;
  if (wants_retake) {
    const std::vector<float> pixels =
        Ltx2ReadFrameDirectory(gen.ref_video_dir, retake_source.height, retake_source.width);
    int64_t cropped = 0;
    const Ltx2LatentVolume encoded = Ltx2ConvVideoEncode(
        im.video_encoder_cfg, im.video_encoder_weights, pixels, im.video_encoder_cfg.in_channels,
        retake_source.frames, retake_source.height, retake_source.width, &cropped);
    // `VideoLatentShape.from_pixel_shape(output_shape).frames`
    // (utils/helpers.py:230-232), which is the same `(frames - 1) / time + 1`
    // the phase loop below derives — computed here from the SOURCE so the
    // conform has something to conform to.
    const int64_t want_latent_frames = (retake_source.frames - 1) / factors.time + 1;
    retake_video_volume =
        Ltx2ConformLatentLength(encoded.data, encoded.channels, encoded.frames,
                                encoded.height * encoded.width, want_latent_frames);
    retake_plan = Ltx2RetakePlanModalities(regenerate_video, regenerate_audio,
                                           /*has_audio_latent=*/false);
    im.trace.retake_conditioned = retake_plan.video_conditioned;
    im.trace.retake_latent_digest = DigestF32(retake_video_volume);
    im.trace.retake_latent_absmax = AbsMax(retake_video_volume);
  }

  prep_retake.Close();
  phase::Scope prep_guiders("generate.guiders");
  // ── THE GUIDERS, and the negative conditioning they ask for (#1092) ───────
  //
  // `create_multimodal_guider_factory(params=..., negative_context=...)` once per
  // stream, before the stage runs (ti2vid_one_stage.py:210-218). Resolved for
  // EVERY phase up front rather than inside the loop, because the negative
  // encode below is a host-side pass over the text tower and must happen once
  // for the whole render if ANY phase asks for it.
  //
  // A phase whose recipe sets no guidance keeps `Ltx2MultiModalGuiderParams`'s
  // own defaults — `cfg 1.0 / stg 0.0 / modality 1.0 / rescale 0.0` — which is
  // exactly `_POSITIVE_ONLY_GUIDER` (denoisers.py:25-28). Only `OneStagePhase`
  // sets real scales, so `distilled_two_stage`, `dfr`, `retake` and `dmd2` run
  // ONE forward per step through the guided seam and combine it with a guider
  // whose every term is zero, which is `SimpleDenoiser`'s output. Upstream
  // selects `SimpleDenoiser` by PIPELINE (distilled.py:266,295) rather than by
  // params; the two agree here because the recipes that select it are exactly
  // the recipes whose guidance is the no-op one.
  struct PhaseGuidance {
    Ltx2MultiModalGuiderParams video;
    Ltx2MultiModalGuiderParams audio;
  };
  std::vector<PhaseGuidance> phase_guidance(recipe.phases.size());
  bool wants_negative = false;
  for (size_t p = 0; p < recipe.phases.size(); ++p) {
    phase_guidance[p].video = recipe.phases[p].video_guidance;
    phase_guidance[p].audio = recipe.phases[p].audio_guidance;
    ApplyGuidanceOverrides(gen.extras, recipe.phases[p], &phase_guidance[p].video,
                           &phase_guidance[p].audio);
    if (phase_guidance[p].video.DoUnconditionalGeneration() ||
        phase_guidance[p].audio.DoUnconditionalGeneration()) {
      wants_negative = true;
    }
  }

  // THE PERTURBED PASSES RUN ON BOTH RESIDENCIES SINCE 2026-08-19, and until then
  // this is where the device arm was refused by name. `Ltx2DitForwardDevice` took
  // no `perturbations` argument, so the `ptb` and `mod` passes there would have
  // run an UNPERTURBED forward — a finite clip whose
  // `stg_scale * (cond - perturbed)` and `(modality_scale - 1) * (cond - mod)`
  // terms are identically zero, indistinguishable from a working render at every
  // output this engine has. The refusal was the right answer to a real gap; the
  // gap is closed (ltx2_device.h, .agents/specs/ltx25-guided-video.md §12) and
  // the refusal is gone with it rather than kept as a safety blanket over a
  // capability that exists.
  //
  // WHAT THAT COSTS, so nobody discovers it as a regression: a render whose
  // guiders carry the model's own defaults (stg 1.0, modality 3.0 —
  // ltx2_pipeline.cpp:947-963) now assembles FOUR passes per step where the
  // device arm previously ran two, which is 2.0x the DiT time. #1375 measured
  // ~162 s per forward on GB10 at 1024x576 with 60 forwards structural; the same
  // render is 120 forwards. Setting the two STG scales to 0.0 and the two
  // modality scales to 1.0 buys that back at the trajectory the device arm had
  // before, and those four extras are documented in docs/USAGE.md.

  // The second half of upstream's ONE `PromptEncoder` call over
  // `[prompt, negative_prompt]` (ti2vid_one_stage.py:166-174). Encoded ONLY when
  // a guider asks: `do_unconditional_generation` is `not isclose(cfg_scale, 1.0)`
  // (guiders.py:275-277), and at 1.0 there is no unconditional forward, so
  // encoding it would be a wasted host-side 12B pass per request.
  std::vector<float> negative_video, negative_audio;
  const float* negative_video_context = nullptr;
  const float* negative_audio_context = nullptr;
  if (wants_negative) {
    if (!im.negative_video_prompt_embeds.empty() && gen.prompt.empty()) {
      // The embeds fallback's own second half. Taken only when the request
      // carries no prompt, which is the same polarity the POSITIVE fallback has
      // above: a typed prompt encodes both halves through the tower.
      if (im.prompt_tokens != context_tokens) {
        Fail("the negative prompt embeds hold " + std::to_string(im.prompt_tokens) +
             " rows and this request's conditioning holds " + std::to_string(context_tokens) +
             "; the guidance delta would subtract tensors that do not correspond");
      }
      negative_video_context = im.negative_video_prompt_embeds.data();
      negative_audio_context = im.negative_audio_prompt_embeds.data();
    } else if (!im.has_encoder) {
      Fail("this render needs an unconditional forward (the video cfg scale is " +
           std::to_string(phase_guidance[0].video.cfg_scale) + " and the audio one is " +
           std::to_string(phase_guidance[0].audio.cfg_scale) +
           "), which needs the NEGATIVE prompt encoded — and no text tower is loaded. The "
           "positive `prompt_embeds_path` fallback carries ONE conditioning pair; supply the "
           "second through '" +
           std::string(kLtx2NegativePromptEmbedsExtra) + "' and '" +
           std::string(kLtx2NegativeAudioPromptEmbedsExtra) +
           "', load with encoder_path, or set '" + std::string(kLtx2VideoCfgScaleExtra) +
           "' and '" + std::string(kLtx2AudioCfgScaleExtra) +
           "' to 1.0, which turns the unconditional pass off (guiders.py:275-277)");
    } else {
      const std::string negative =
          VideoExtra(gen.extras, kLtx2NegativePromptExtra, recipe.negative_prompt);
      if (negative.empty()) {
        Fail("this render needs a negative prompt and neither the '" +
             std::string(kLtx2NegativePromptExtra) +
             "' extra nor the recipe carries one. An EMPTY negative prompt is not the same as no "
             "CFG: it still encodes and still steers, and upstream's CLI always supplies "
             "`DEFAULT_NEGATIVE_PROMPT` (utils/args.py:937-946)");
      }
      if (!recipe.allow_negative_prompt) {
        Fail("this recipe takes no negative prompt (`prompts_to_encode` is `[prompt]` alone), so "
             "a guider asking for the unconditional forward is a contradiction rather than a "
             "request this engine can serve");
      }
      vt::Queue text_queue{vt::Device{vt::DeviceType::kCPU, 0}, nullptr};
      const Ltx2PromptConditioning encoded = Ltx2EncodePromptToConditioning(
          *im.tower, *im.tokenizer, im.gemma_ids, im.caption_projections, im.feature_cfg,
          negative, text_queue);
      negative_video = encoded.conditioning.video;
      negative_audio = encoded.conditioning.audio;
      if (encoded.seq != context_tokens) {
        // Upstream's two encodings come from ONE tokenization of a two-element
        // list, so they share a padded width by construction. A mismatch means
        // the two ran different geometries and the guidance delta would subtract
        // tensors that do not correspond.
        Fail("the negative prompt encoded to " + std::to_string(encoded.seq) +
             " context rows and the prompt to " + std::to_string(context_tokens) +
             "; upstream encodes both in one call and they cannot differ");
      }
      if (im.has_connector) {
        const Ltx2ConnectorEmbeddings through =
            RunConnector(SafetensorsFile::Open(im.params.dit_path), im.video_connector_cfg,
                         im.audio_connector_cfg, encoded.conditioning.video,
                         encoded.conditioning.audio, encoded.conditioning.additive_mask,
                         context_tokens);
        negative_video = through.video;
        negative_audio = through.audio;
      }
      negative_video_context = negative_video.data();
      negative_audio_context = negative_audio.data();
    }
  }

  prep_guiders.Close();
  for (int64_t phase_index = 0; phase_index <= last_phase; ++phase_index) {
    const Ltx2PhaseRecipe& phase = recipe.phases[static_cast<size_t>(phase_index)];
    // W0: one set of leaves PER RECIPE PHASE. The two-stage recipe renders its
    // stages at different resolutions, so a table that summed them would hide
    // exactly the term the campaign is looking for.
    ::vllm::multimodal::phase::Scope phase_prepare("phase.prepare");

    // THE PER-PHASE ADAPTER SET (row LTX25-PHASE-LORA, issue #1118). Upstream
    // hands each `DiffusionStage` its own `loras=` argument
    // (a2vid_two_stage.py:107 against :114) and pays for it with a second
    // resident weight set; this engine holds ONE DiT and moves it between the
    // two states here, re-materializing only the tensors an adapter targets.
    //
    // BEFORE any conditioning, encode or forward of this phase, so no work is
    // ever paid against weights the phase did not ask for. A no-op when the
    // load supplied no adapter, and a no-op when the DiT is already in the
    // requested state — so a one-stage recipe and every recipe that predates
    // this field cost nothing.
    //
    // WHAT A TWO-STAGE RENDER PAYS IS TWO REBINDS, not one, and the count is
    // written out because it is the wall-clock half of the trade the row's spec
    // accepted. `a2vid_two_stage` loads FUSED, phase 0 asks `kNoAdapters` and
    // rebinds off, phase 1 asks `kAllAdapters` and rebinds back on; the DiT is
    // left fused, so the NEXT render pays the same two. Each one re-opens the
    // adapter and reads every A/B factor pair (`Ltx2LoraAdapter::Open` ->
    // `ReadFactorAsBf16`), and the shipped distilled adapter is 8.9 GB. That
    // cost is UNMEASURED on real weights; the row claims no wall-clock result
    // and a later perf row owns the number.
    //
    // The emptiness test is HERE as well as inside the rebind so that a load
    // with no adapter does not re-open the checkpoint once per phase to be told
    // there is nothing to do.
    {
      const bool want_fused = phase.loras == Ltx2PhaseLoraScope::kAllAdapters;
      if (!im.dit_options.loras.empty() && want_fused != (im.dit.lora_fused_tensors > 0)) {
        Ltx2RebindDitLoras(im.on_device ? &*im.queue : nullptr,
                           SafetensorsFile::Open(im.params.dit_path), im.dit_options,
                           want_fused, im.dit);
      }
    }

    const int64_t phase_h = height / phase.spatial_downscale;
    const int64_t phase_w = width / phase.spatial_downscale;

    Ltx2VideoLatentShape vshape;
    vshape.channels = im.dit.params.in_channels;
    vshape.frames = (frames - 1) / factors.time + 1;
    vshape.height = phase_h / factors.height;
    vshape.width = phase_w / factors.width;
    if (vshape.frames < 1 || vshape.height < 1 || vshape.width < 1) {
      Fail("phase '" + phase.name + "' resolves a latent of " + std::to_string(vshape.frames) +
           "x" + std::to_string(vshape.height) + "x" + std::to_string(vshape.width) +
           " from " + std::to_string(frames) + " frames at " + std::to_string(phase_w) + "x" +
           std::to_string(phase_h) + "; the VAE downscales by " +
           std::to_string(factors.time) + "x" + std::to_string(factors.height) + "x" +
           std::to_string(factors.width) + " so the request is below one latent cell");
    }

    Ltx2AudioLatentShape ashape;
    ashape.channels = kAudioLatentChannels;
    ashape.mel_bins = kAudioLatentMelBins;
    {
      // `AudioLatentShape.from_duration` (types.py:164-181).
      const Ltx2AudioPatchifierParams ap;
      const double latents_per_second = static_cast<double>(ap.sample_rate) /
                                        static_cast<double>(ap.hop_length) /
                                        static_cast<double>(ap.audio_latent_downsample_factor);
      ashape.frames = static_cast<int64_t>(
          std::llround(static_cast<double>(frames) / fps * latents_per_second));
    }
    if (ashape.frames < 1) Fail("the audio latent resolved to zero frames");

    // ── the input transform (ltx2_recipes.py:38) ────────────────────────────
    std::vector<float> video_initial;
    if (wants_retake) {
      // `ModalitySpec(initial_latent=initial_video_latent)` (retake.py:273).
      // The `retake` recipe is one phase with no input transform, so this is
      // the only producer of `video_initial` on this path; the refusal at load
      // makes any other recipe unreachable from a retake request.
      if (retake_video_volume.empty()) Fail("the retake source produced no latent");
      video_initial = retake_video_volume;
    }
    if (phase.input_transform == Ltx2PhaseInputTransform::kSpatialUpsample) {
      if (video_latent_volume.empty()) {
        Fail("phase '" + phase.name +
             "' asks for the spatial-upsample input transform but no earlier phase produced a "
             "latent");
      }
      if (!im.has_upsampler) {
        Fail("phase '" + phase.name +
             "' needs the latent spatial x2 upsampler, and no 'upsampler_path' load extra was "
             "supplied. Refusing rather than skipping the phase: its 3-step refinement is what "
             "makes the upscaled latent valid, and running the decode on the half-resolution "
             "latent instead would return a smaller clip that looks like a completed request. "
             "Supply 'upsampler_path', or stop before this phase with '" +
             std::string(kLtx2MaxPhaseExtra) + "'.");
      }
      // The phase wants the SPATIAL x2 upsampler. The temporal x2 arm is a
      // different model with the same class name and the same tensor NAMES
      // (`upsampler.0.*`), so pointing 'upsampler_path' at
      // `ltx-2.5-latent-temporal-upscaler-x2-bf16-1.0.safetensors` loads and runs
      // cleanly and returns `[c, 2f-1, h, w]` where this phase needs
      // `[c, f, 2h, 2w]`. The shape check below would catch it, but it would
      // report a mismatch and leave the caller guessing; naming the swap here is
      // the difference between "you gave me the wrong checkpoint" and "something
      // is 3 frames short". Ported and gated, not driven — see
      // .agents/specs/ltx25-temporal-upsampler.md section 7.
      //
      // `&& !spatial_upsample` IS LOAD-BEARING. This guard used to test
      // `temporal_upsample` alone, which every BOTH-flags config also satisfies,
      // so it fired by implication over the same variable and told the caller who
      // supplied a genuine SPATIOTEMPORAL checkpoint that they had handed over the
      // temporal one — wrong on both counts, and pointing them at the arm they
      // already had. It also shadowed the ledger refusal at
      // `ltx2_upsampler.cpp:465`, which names the spatiotemporal arm and was
      // therefore unreachable from any request. Narrowed here so a both-flags
      // config falls THROUGH to that refusal. Gated by test_ltx2_video's
      // "a SPATIOTEMPORAL upsampler checkpoint is refused as SPATIOTEMPORAL".
      if (im.upsampler_cfg.temporal_upsample && !im.upsampler_cfg.spatial_upsample) {
        Fail("phase '" + phase.name +
             "' needs the latent SPATIAL x2 upsampler, but the checkpoint at 'upsampler_path' "
             "declares temporal_upsample=true, i.e. it is the TEMPORAL x2 upsampler. That arm "
             "upsamples the frame axis and drops the first frame "
             "(model/upsampler/model.py:68-71, 109-113); no phase of any recipe this engine "
             "serves consumes it, because its only upstream consumer is DFRPipeline's rounds "
             "loop, which is not ported. Supply the spatial upsampler "
             "('ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors').");
      }
      // W0 repair (#1010 review): #1010 names "any latent-upsampler stage" and
      // this one had no name — it ran inside `phase.prepare`, whose name says
      // nothing about it. NESTED inside that leaf for the same reason the audio
      // split is: the decomposition is readable and the sum does not move.
      // Reached only by a recipe phase whose input transform is the spatial
      // upsample, i.e. stage 2 of the two-stage recipes.
      const ::vllm::multimodal::phase::Scope upsample_phase("phase.upsample_latent");
      Ltx2LatentVolume in;
      in.batch = 1;
      in.channels = video_lc;
      in.frames = video_lf;
      in.height = video_lh;
      in.width = video_lw;
      in.data = video_latent_volume;
      // `upsample_video` (upsampler/model.py:129-143): un-normalize by the video
      // ENCODER's per-channel statistics, upsample, re-normalize. Those live in
      // the VAE checkpoint, not in the upsampler's.
      const Ltx2LatentVolume up = Ltx2UpsampleVideoLatent(
          im.upsampler_cfg, im.upsampler_weights, in,
          im.video_weights.Get("per_channel_statistics.std-of-means"),
          im.video_weights.Get("per_channel_statistics.mean-of-means"));
      if (up.channels != vshape.channels || up.frames != vshape.frames ||
          up.height != vshape.height || up.width != vshape.width) {
        Fail("the upsampled latent is " + std::to_string(up.channels) + "x" +
             std::to_string(up.frames) + "x" + std::to_string(up.height) + "x" +
             std::to_string(up.width) + " but phase '" + phase.name + "' needs " +
             std::to_string(vshape.channels) + "x" + std::to_string(vshape.frames) + "x" +
             std::to_string(vshape.height) + "x" + std::to_string(vshape.width));
      }
      video_initial = up.data;

      // THE SLOTS TAKE THE SAME UPSAMPLER (#986). `dfr_pipeline.py:348` runs the
      // stage-1 slots through `self.upsampler` — the SPATIAL one, the same
      // object and the same call as the video latent on the line after it
      // (:349) — before stage 2 uses them as `initial_keyframes` (:364).
      //
      // Each slot is one latent frame, and they are upsampled as a K-frame
      // volume here because that is what upstream hands over: `self.upsampler`
      // takes the whole `(B, C, K, H, W)` tensor. That is safe for the SPATIAL
      // arm in a way it would not be for a decode — the operator is
      // convolutional in H and W and the temporal axis only rides along, which
      // is why upstream can upsample slots together and must still DECODE them
      // apart (types.py:269-272).
      if (slot_keyframes.frames > 0) {
        const Ltx2LatentVolume up_slots = Ltx2UpsampleVideoLatent(
            im.upsampler_cfg, im.upsampler_weights, slot_keyframes,
            im.video_weights.Get("per_channel_statistics.std-of-means"),
            im.video_weights.Get("per_channel_statistics.mean-of-means"));
        if (up_slots.height != vshape.height || up_slots.width != vshape.width ||
            up_slots.channels != vshape.channels ||
            up_slots.frames != static_cast<int64_t>(slot_positions.size())) {
          Fail("the upsampled generated keyframe slots are " +
               std::to_string(up_slots.channels) + "x" + std::to_string(up_slots.frames) + "x" +
               std::to_string(up_slots.height) + "x" + std::to_string(up_slots.width) +
               " but phase '" + phase.name + "' needs " + std::to_string(vshape.channels) + "x" +
               std::to_string(slot_positions.size()) + "x" + std::to_string(vshape.height) + "x" +
               std::to_string(vshape.width) +
               ". The slots take the SAME spatial upsampler as the video latent "
               "(dfr_pipeline.py:348-349), so a disagreement here means the two took different "
               "paths.");
        }
        slot_keyframes = up_slots;
      }
    }

    // ── build the two states (create_noised_state, helpers.py:428-445) ───────
    //
    // `target_tokens` is `patchifier.get_token_count(target_shape)` — the count
    // of the TARGET GRID, which is fixed for this phase. `video.tokens` starts
    // equal to it and is GROWN by any appending conditioning item. Row
    // LTX25-TOKEN-APPEND (#930) split the two, and the two places that must keep
    // reading the target rather than the grown count are the sigma schedule
    // (schedulers.py:32-39 reads the unpatchified target at :32 and turns it into
    // the shift at :39, and
    // the pipelines compute sigmas before the state exists at all) and the trim
    // at the bottom of the loop (`clear_conditioning`, tools.py:101).
    const int64_t target_tokens = Ltx2VideoTokenCount(vshape, 1);
    StreamState video;
    video.width = vshape.channels;  // patch_size 1 (VideoLatentPatchifier(1))
    video.tokens = target_tokens;
    {
      std::vector<float> volume(static_cast<size_t>(vshape.channels) *
                                static_cast<size_t>(vshape.frames) *
                                static_cast<size_t>(vshape.height) *
                                static_cast<size_t>(vshape.width));
      if (!video_initial.empty()) {
        if (video_initial.size() != volume.size()) Fail("the initial video latent is the wrong size");
        volume = video_initial;
      }
      video.latent = Ltx2VideoPatchify(volume.data(), vshape, 1);
      video.clean = video.latent;
      video.mask.assign(static_cast<size_t>(video.tokens), 1.0F);
      // ── RETAKE: TemporalRegionMask (noise_mask_cond.py:23-45) ─────────────
      //
      // `state.denoise_mask.copy_(mask_val)` (:44) REPLACES the all-ones
      // `create_initial_state` just wrote (ltx-core/tools.py:158-161), which is
      // why this sits here rather than multiplying into the vector.
      //
      // Nothing else in the loop changes, and that is the design. Every consumer
      // of `mask` already broadcasts it: `ApplyGaussianNoise` leaves the latent
      // at `clean` where the mask is 0, `TimestepsFromMask` yields per-token
      // timestep 0 there, and `PostProcessLatent` blends `clean` back every
      // step. `video.clean = video.latent` above is what makes that blend
      // restore the SOURCE rather than zeros — `clean_latent =
      // initial_latent.clone()` (ltx-core/tools.py:156).
      if (wants_retake && retake_plan.video_conditioned) {
        // `causal_fix` TRUE, which is the CALL SITE's default
        // (noise_mask_cond.py:33 reads `getattr(latent_tools, "causal_fix",
        // True)`) and NOT `get_pixel_coords`'s own, which is False
        // (patchifiers.py:140). Taking the function's would move every boundary
        // by `time - 1` = 7 pixel frames and still render.
        video.mask = Ltx2TemporalRegionMaskVideo(vshape, /*patch_size=*/1, factors, fps,
                                                 retake_start, retake_end, /*causal_fix=*/true);
        if (static_cast<int64_t>(video.mask.size()) != video.tokens) {
          Fail("the temporal region mask is " + std::to_string(video.mask.size()) +
               " tokens and the video stream is " + std::to_string(video.tokens));
        }
        int64_t inside = 0;
        for (const float value : video.mask) {
          if (value != 0.0F) ++inside;
        }
        im.trace.retake_masked_tokens = inside;
        im.trace.retake_total_tokens = video.tokens;
      } else if (wants_retake && retake_plan.video_frozen) {
        // `frozen=not regenerate_video` with an empty conditioning list
        // (retake.py:274, :271-272). The zeroed mask is the first half; the
        // scalar `Modality.sigma` is the second and is applied at the forward,
        // exactly as the audio side already does — upstream's parenthesis at
        // utils/types.py:104-106 says the two are different.
        video.mask.assign(static_cast<size_t>(video.tokens), 0.0F);
      }
      // tools.py:184 — `create_initial_state` returns the state with
      // `keyframes_mask=self._first_frame_keyframes_mask(state)` ALWAYS, on the
      // same line that builds it. Not conditioned on `wants_image`, not
      // conditioned on any keyframe: the marker is a fact about the causal
      // encoder's first latent frame, which spans a single pixel frame while
      // every later one spans `temporal_scale_factor`.
      video.keyframes_mask = Ltx2FirstFrameKeyframesMask(vshape, /*patch_size=*/1);
      const std::vector<int64_t> bounds = Ltx2VideoPatchBounds(vshape, 1);
      const std::vector<int64_t> pixels = Ltx2PixelCoords(bounds, 1, video.tokens, factors, true);
      // `positions = get_pixel_coords(...).float()` then
      // `positions[:, 0, ...] /= fps` (tools.py:169-174). The division is f32
      // upstream, so it is done in float here and only then widened to the
      // double the DiT's `positions` field takes.
      video.positions.resize(pixels.size());
      for (size_t i = 0; i < pixels.size(); ++i) {
        const float value = static_cast<float>(pixels[i]);
        const bool temporal = i < static_cast<size_t>(video.tokens) * 2;
        video.positions[i] = temporal ? static_cast<double>(value / static_cast<float>(fps))
                                      : static_cast<double>(value);
      }
    }

    // Hoisted out of the state block below because the FORWARD needs it too:
    // upstream's `frozen` sets the scalar `Modality.sigma` as well as the mask.
    const bool audio_frozen = !a2v_audio_volume.empty();
    StreamState audio;
    audio.width = ashape.channels * ashape.mel_bins;
    audio.tokens = ashape.frames;
    {
      std::vector<float> volume(static_cast<size_t>(ashape.channels) *
                                static_cast<size_t>(ashape.frames) *
                                static_cast<size_t>(ashape.mel_bins));
      if (phase_index > 0) {
        // Stage 2 re-noises stage 1's AUDIO latent rather than starting from
        // zeros (distilled.py:307-311). Dropping that carry-over regenerates the
        // soundtrack from scratch with a 3-step schedule, which produces audio.
        if (audio_latent_volume.size() != volume.size()) {
          Fail("the audio latent changed size between phases");
        }
        volume = audio_latent_volume;
      }
      // ── audio-to-video: the supplied take, FROZEN (#922) ──────────────────
      //
      // `ModalitySpec(context=..., frozen=True, noise_scale=0.0,
      // initial_latent=encoded_audio_latent)` — a2vid_two_stage.py:251-256 for
      // stage 1 and :291-296 for stage 2, identical on both. It replaces the
      // carry-over above rather than adding to it: upstream hands the SAME
      // encoded tensor to both stages, so phase 2 must not inherit phase 1's
      // audio state.
      if (audio_frozen) {
        if (a2v_audio_volume.size() != volume.size()) {
          Fail("the encoded audio latent is " + std::to_string(a2v_audio_volume.size()) +
               " values and phase '" + phase.name + "' wants " + std::to_string(volume.size()));
        }
        volume = a2v_audio_volume;
      }
      audio.latent = Ltx2AudioPatchify(volume.data(), ashape);
      audio.clean = audio.latent;
      // `frozen=True` "zeros the denoise mask and marks the resulting
      // LatentState so Modality.sigma is forced to 0 (not only per-token
      // timesteps)" — upstream's own words at utils/types.py:104-106. The
      // zeroed mask is the whole of the first half here, and it is load-bearing
      // three times over, because every consumer of `mask` already broadcasts
      // it: `ApplyGaussianNoise` leaves the latent at `clean`, so
      // `noise_scale=0.0` needs no separate branch; `TimestepsFromMask` yields
      // per-token timestep 0, so the DiT sees the audio as clean conditioning;
      // and the STEP cannot move the latent either.
      //
      // That third one holds by a DIFFERENT argument on each stepper arm, and
      // "`PostProcessLatent` blends it back every step" — which this comment
      // used to say — is true of only one of them. On `kEulerAncestral` the
      // stepped latent IS passed back through `PostProcessLatent`, which
      // restores `clean` wherever the mask is 0. The plain `kEuler` arm never
      // calls it, and does not need to: `a_denoised` is itself post-processed,
      // so on a frozen stream `a_denoised == clean == latent`, the Euler
      // derivative `d = (x - denoised) / sigma` is exactly 0, and
      // `x + (sigma_next - sigma) * d` returns `x` unchanged. Two arms, two
      // reasons, one invariant — and the weaker one is the one that had to be
      // written down, because a reader who checks the strong claim against
      // `kEuler` finds no `PostProcessLatent` there and concludes the freeze
      // leaks.
      //
      // The SECOND half — the scalar `Modality.sigma` —
      // is not expressible through the mask and is applied at the forward
      // below; upstream's parenthesis says exactly that the two are different.
      audio.mask.assign(static_cast<size_t>(audio.tokens), audio_frozen ? 0.0F : 1.0F);
      const std::vector<float> timings = Ltx2AudioPatchTimings(ashape, Ltx2AudioPatchifierParams{});
      audio.positions.assign(timings.begin(), timings.end());
    }

    im.trace.audio_conditioned = !a2v_audio_volume.empty();
    im.trace.audio_tokens = audio.tokens;

    // ── the image conditioning (issue #644) ─────────────────────────────────
    //
    // BEFORE THE NOISER AND AFTER THE STATE, which is upstream's order
    // (`create_noised_state`, helpers.py:428-445: initial state, THEN the
    // conditioning items, THEN the noiser) and is not interchangeable: the
    // item writes ONLY `clean_latent` and `denoise_mask` (latent_cond.py:38-39)
    // and the noiser is what composes them into the noisy tensor
    // (components/noisers.py:31-34). Applying it afterwards leaves the
    // conditioned tokens pinned to NOISE, with an identical clean tensor and an
    // identical mask — so nothing but the noised latent itself can see it.
    //
    // PER PHASE, and encoded per phase, because the two-stage recipe renders its
    // stages at DIFFERENT resolutions (`phase.spatial_downscale`) and upstream
    // passes each stage's own height/width to `combined_image_conditionings`,
    // whose `height` / `width` are per-call parameters (helpers.py:274-275) that
    // distilled.py fills differently per stage: `stage_1_w, stage_1_h = width //
    // 2, height // 2` at :251 passed at :255-256, against the full-resolution
    // `height` / `width` at :285-286. Conditioning stage 1 only would let stage 2 re-noise
    // the pinned frame away; conditioning stage 2 with stage 1's latent would
    // place a half-resolution image into a full-resolution grid.
    // ONE ITERATION of upstream's `images` loop (helpers.py:283-291): load,
    // preprocess to THIS phase's height and width, encode. Shared by both arms
    // because upstream shares it — the branch is on `frame_idx`, below, and it
    // is the only thing that differs between them.
    auto encode_conditioning_image = [&](const char* label,
                                         const std::string& bytes) -> Ltx2LatentVolume {
      const std::vector<float> pixels =
          Ltx2LoadImageAndPreprocess(label, bytes, phase_h, phase_w, image_crf);
      int64_t cropped = 0;
      const Ltx2LatentVolume encoded = Ltx2ConvVideoEncode(
          im.video_encoder_cfg, im.video_encoder_weights, pixels,
          im.video_encoder_cfg.in_channels, /*frame_count=*/1, phase_h, phase_w, &cropped);
      if (encoded.frames != 1) {
        Fail("the video VAE encoder returned " + std::to_string(encoded.frames) +
             " latent frames for a single image; both arms of "
             "`combined_image_conditionings` place exactly one "
             "(ltx-pipelines/utils/helpers.py:294-300)");
      }
      // The REPLACE arm needs this to hold because upstream raises
      // ConditioningError otherwise (latent_cond.py:25-30). The APPEND arm does
      // not — `VideoConditionByKeyframeIndex` derives its own shape from the
      // keyframe tensor (keyframe_cond.py:41-44) and never compares it to the
      // target — so for that arm this is a consistency assertion rather than a
      // capability limit, and it cannot falsely fire: the preprocess above
      // targets this phase's own height and width, so a disagreement here means
      // the encoder's spatial factor and VIDEO_SCALE_FACTORS disagree, which
      // would place the keyframe's tokens at the wrong RoPE positions.
      if (encoded.channels != vshape.channels || encoded.height != vshape.height ||
          encoded.width != vshape.width) {
        Fail(std::string("the encoded ") + label + " is " + std::to_string(encoded.channels) +
             "x" + std::to_string(encoded.height) + "x" + std::to_string(encoded.width) +
             " but phase '" + phase.name + "' needs " + std::to_string(vshape.channels) + "x" +
             std::to_string(vshape.height) + "x" + std::to_string(vshape.width) +
             ". Upstream raises ConditioningError on exactly this "
             "(conditioning/types/latent_cond.py:25-30): the encoder's spatial factor and the "
             "pipeline's VIDEO_SCALE_FACTORS must agree, and they do not.");
      }
      return encoded;
    };

    if (wants_first_frame) {
      const Ltx2LatentVolume encoded = encode_conditioning_image("first_frame", image_bytes);

      // WHICH ITEM FRAME 0 TAKES IS THE RECIPE'S TO SAY, and it is the ONE line
      // on which upstream's two image-conditioning builders differ (row
      // LTX25-KEYFRAME-INTERP, #1096):
      //
      //   combined_image_conditionings (helpers.py:295-300)
      //       frame_idx == 0 -> VideoConditionByLatentIndex. REPLACES tokens
      //       that already exist; the token count never changes.
      //   image_conditionings_by_adding_guiding_latent (helpers.py:343-367)
      //       NO branch. Frame 0 takes VideoConditionByKeyframeIndex like every
      //       other frame, and APPENDS a latent frame of tokens.
      //
      // A `pipeline_kind` string test here would be one more place for the next
      // recipe on the second builder to be missed, so the flag rides the recipe
      // — see `Ltx2ImageConditioningBuilder`.
      //
      // NOTHING ABOUT THE RENDER'S SHAPE CAN SEE THIS. Both arms condition on
      // the same image and both return a clip of the right size, the right frame
      // count and the right sample rate; the only observable is the sequence
      // LENGTH the DiT ran over, which is `im.trace.video_tokens`.
      const bool frame0_appends =
          recipe.image_conditioning == Ltx2ImageConditioningBuilder::kAddGuidingLatent;
      // The sequence length BEFORE this item, so the digest below can be taken
      // from the tokens the append actually grew. On the replace arm it is also
      // the length after, which the assertion under `frame0_appends` states.
      const int64_t before_first_frame = video.tokens;

      Ltx2LatentState state = ToLatentState(video, /*pos_dims=*/3);
      if (frame0_appends) {
        // `frame_idx = 0` is not a formality: `VideoConditionByKeyframeIndex`
        // offsets its positions by `frame_idx` (keyframe_cond.py:52), so this
        // argument alone decides WHERE IN TIME the opening keyframe lands. It is
        // gated below, on a temporal window recomputed from `fps`, because
        // nothing else can see it — the sibling arm found the same hole with
        // mutation M10 and carries the same shape of check.
        //
        // `causal_fix = true` is upstream's value for this item and it is INERT
        // at this call site. MEASURED, so the next reader does not go looking
        // for the gate that cannot exist: `Ltx2ConditionVideoByKeyframe` gates
        // the fix on `frame_idx == 0` (keyframe_cond.py:49,
        // `latent_tools.causal_fix if self.frame_idx == 0 else False`), so the
        // gate is OPEN here and the fix is applied — and then the
        // `num_pixel_frames == 1` narrow at `:56-57` overwrites the temporal END
        // the fix moved, while the temporal START clamps to 0 either way for a
        // keyframe whose latent depth is 1 (`max(0 + 1 - time, 0)`, and
        // `encoded.frames != 1` is refused above). Flipping this argument moves
        // 0 of 48 position values on a probe of the same shapes; the fix becomes
        // observable only at `num_pixel_frames != 1`, which no arm here passes.
        // The `frame_idx == 0` gate itself is what carries the risk, and it is
        // gated on the seam in `test_ltx2_vae` at a shape where it shows.
        Ltx2ConditionVideoByKeyframe(&state, encoded, /*patch_size=*/1, factors, fps,
                                     /*frame_idx=*/0, image_strength,
                                     /*num_pixel_frames=*/1, /*causal_fix=*/true);
      } else {
        Ltx2ConditionVideoByLatentIndex(&state, vshape, /*patch_size=*/1, encoded, image_strength,
                                        /*latent_idx=*/0);
      }
      FromLatentState(state, &video);

      // The witness, taken from the TOKENS THAT WERE WRITTEN rather than from
      // `encoded` — and the difference is not cosmetic. Digesting the encoder's
      // output would answer "was an image encoded", which stays true of a build
      // that encodes an image and then never places it: the render would be
      // unconditioned and every field here would look healthy. Digesting the
      // conditioned slice of the clean latent answers "did those tokens reach
      // the state", which is the question. Filled on the LAST phase, so it
      // describes the conditioning the finished latent carries.
      //
      // IT IS STILL A CHANGE DETECTOR, not a value gate — the same limit the two
      // prompt digests carry. What the placed tokens should NUMERICALLY be is
      // gated against executed upstream in `test_ltx2_image_cond`, which drives
      // these very functions; MEASURED, because a mutation that moved the
      // composition inside this loop and left `test_ltx2_video` green is how
      // this comment came to be here.
      const int64_t placed = Ltx2VideoTokenCount({1, vshape.channels, 1, vshape.height,
                                                  vshape.width},
                                                 1);
      // AND WHERE THEY LANDED DEPENDS ON THE ARM. The replace arm writes latent
      // frame 0, which is the FRONT of the clean tensor; the append arm writes
      // the tokens it just added, which are the TAIL. Reading the front on the
      // append arm would digest unconditioned tokens and report a healthy
      // conditioning for a state the keyframe never reached — exactly the
      // failure the paragraph above says this field exists to prevent.
      const ptrdiff_t first = frame0_appends
                                  ? static_cast<ptrdiff_t>(before_first_frame * video.width)
                                  : 0;
      const std::vector<float> written(
          video.clean.begin() + first,
          video.clean.begin() + first + static_cast<ptrdiff_t>(placed * video.width));
      // AND THE SLICE IS THE ONE THE IMAGE LANDED IN. `first` is the only thing
      // that decides which tokens the three fields below describe, and it was
      // ungated: forcing it to 0 on the append arm made the digest describe the
      // untouched FRONT of the sequence, and both binaries stayed green — which
      // is the exact failure the paragraph above says these fields exist to
      // prevent, arriving through the slice rather than through the conditioning.
      //
      // Recomputed from `encoded`, not from `first`, so the two are independent
      // expressions. Both conditioning items write the SAME bytes — the
      // patchified conditioning volume — the replace arm at latent frame 0
      // (latent_cond.py:38-39) and the append arm at the tail
      // (keyframe_cond.py:82) — so one expectation serves both, and a slice that
      // names any other window reds by value rather than by shape.
      Ltx2VideoLatentShape placed_shape = vshape;
      placed_shape.frames = encoded.frames;
      const std::vector<float> expected =
          Ltx2VideoPatchify(encoded.data.data(), placed_shape, /*patch_size=*/1);
      VT_CHECK(written == expected,
               "ltx2 video: `image_digest` and `image_absmax` must describe the tokens the "
               "conditioning item WROTE. The slice this arm read back does not hold the "
               "patchified conditioning volume, so the trace reports a healthy conditioning for "
               "a window the image never reached");
      im.trace.image_tokens = placed;
      im.trace.image_digest = DigestF32(written);
      im.trace.image_absmax = AbsMax(written);

      // THE ARM DID WHAT ITS NAME SAYS. Asserted rather than assumed for the
      // reason the last-frame arm below asserts the same thing: a build whose
      // append did not grow the sequence leaves buffers longer than the count
      // that describes them, and the DiT then reads a prefix, renders a
      // plausible clip and never mentions the keyframe. The replace polarity is
      // asserted too, because the two arms share one digest slice above and that
      // slice is only correct while each arm moves the count the way this says.
      if (frame0_appends) {
        VT_CHECK(video.tokens == before_first_frame + placed,
                 "ltx2 video: this recipe takes `image_conditionings_by_adding_guiding_latent` "
                 "(ltx-pipelines/utils/helpers.py:343-367), whose frame-0 item APPENDS "
                 "(keyframe_cond.py:79-82), and the sequence did not grow by one latent frame "
                 "of tokens");

        // ...AND IT LANDED ON PIXEL FRAME 0, which is the whole content of
        // `frame_idx` and which nothing above can see. MEASURED, on this arm,
        // exactly as the sibling last-frame arm measured it: changing
        // `frame_idx` from 0 to 3 left both binaries GREEN, because an opening
        // keyframe pinned to the wrong pixel frame appends the same number of
        // tokens carrying the same content and only sits somewhere else in time.
        //
        // The window is recomputed from `fps` alone and never read back from the
        // argument, so the two are independent expressions. Upstream offsets in
        // integer PIXEL space and then divides the temporal axis by fps
        // (keyframe_cond.py:52-59), and `num_pixel_frames = 1` narrows the end to
        // `start + 1` BEFORE that division (`:56-57`) — so the first appended
        // token spans exactly `[0, 1/fps)`. Asserting the END as well as the
        // START is what keeps this check reading the composition rather than the
        // offset alone: it is the one place `num_pixel_frames` is observable.
        const double want_t0 = 0.0;
        const double want_t1 = static_cast<double>(static_cast<float>(1.0 / fps));
        const double got_t0 = video.positions[static_cast<size_t>(before_first_frame * 2)];
        const double got_t1 = video.positions[static_cast<size_t>(before_first_frame * 2 + 1)];
        VT_CHECK(std::abs(got_t0 - want_t0) <= 1e-5 &&
                     std::abs(got_t1 - want_t1) <= 1e-5 * std::max(1.0, std::abs(want_t1)),
                 "ltx2 video: the first-frame keyframe's appended tokens must span pixel frame 0, "
                 "i.e. [" +
                     std::to_string(want_t0) + ", " + std::to_string(want_t1) +
                     "), and the first appended token spans [" + std::to_string(got_t0) + ", " +
                     std::to_string(got_t1) +
                     "). A keyframe that appends the right number of tokens at the wrong TIME "
                     "renders a clip of the right length that pins the image to the wrong end");
      } else {
        VT_CHECK(video.tokens == before_first_frame,
                 "ltx2 video: `combined_image_conditionings` sends frame 0 to "
                 "`VideoConditionByLatentIndex` (helpers.py:295-300), which REPLACES tokens that "
                 "already exist (latent_cond.py:38-39), so the sequence length must not move");
      }
    }

    // ── the LAST-frame keyframe (row LTX25-TOKEN-APPEND, issue #930) ────────
    //
    // The other branch of the same upstream loop: anything but `frame_idx == 0`
    // takes `VideoConditionByKeyframeIndex` (helpers.py:301-305), which APPENDS
    // rather than replaces. `frame_idx` is a PIXEL frame, so the last frame of
    // the output is `frames - 1`; `num_pixel_frames` stays at upstream's default
    // of 1, which is what narrows the appended tokens' temporal extent to
    // `[start, start + 1)` instead of the VAE-scaled range
    // (keyframe_cond.py:53-56).
    //
    // AFTER the first-frame arm, because upstream applies conditioning items in
    // list order (`state_with_conditionings`, helpers.py:448-458) and both arms
    // can be supplied at once — `--image` is repeatable. Order matters here for
    // a reason a shape check cannot see: the appended tokens land at the END of
    // the sequence, and `clear_conditioning` trims from the end, so an item that
    // appended BEFORE a replace would still be trimmed correctly while an item
    // that appended before another append would swap their positions.
    if (wants_last_frame) {
      const Ltx2LatentVolume encoded = encode_conditioning_image("last_frame", last_frame_bytes);

      // THE SEQUENCE LENGTH BEFORE THIS ITEM, and it is not `target_tokens`.
      // Every assertion below locates this item's tokens, and the only thing
      // that knows where they start is the count at the moment of the append —
      // NOT the target grid, which is what stands before the FIRST append and
      // nothing after it. Reading the target here was correct only while this
      // arm was the first append, and row LTX25-KEYFRAME-INTERP put the
      // first-frame arm's own append in front of it on the
      // `image_conditionings_by_adding_guiding_latent` recipes: with both ends
      // pinned, `positions[target_tokens * 2]` named the FIRST frame's keyframe
      // at temporal 0 and the check below aborted the render (#1219).
      const int64_t before_last_frame = video.tokens;

      Ltx2LatentState state = ToLatentState(video, /*pos_dims=*/3);
      Ltx2ConditionVideoByKeyframe(&state, encoded, /*patch_size=*/1, factors, fps,
                                   /*frame_idx=*/frames - 1, image_strength,
                                   /*num_pixel_frames=*/1, /*causal_fix=*/true);
      FromLatentState(state, &video);

      // THE SEQUENCE GREW, and every consumer below reads `video.tokens` rather
      // than the target count. Asserted rather than assumed, because a converter
      // that dropped the grown count would leave a state whose buffers are
      // longer than the count that describes them — and the DiT would then read
      // a prefix, render a plausible clip, and never mention the keyframe.
      VT_CHECK(video.tokens > before_last_frame,
               "ltx2 video: a keyframe conditioning must APPEND tokens "
               "(keyframe_cond.py:79-82) and this one left the sequence length unchanged");
      VT_CHECK(static_cast<int64_t>(video.latent.size()) == video.tokens * video.width &&
                   static_cast<int64_t>(video.clean.size()) == video.tokens * video.width &&
                   static_cast<int64_t>(video.mask.size()) == video.tokens &&
                   static_cast<int64_t>(video.keyframes_mask.size()) == video.tokens &&
                   static_cast<int64_t>(video.positions.size()) == 3 * video.tokens * 2,
               "ltx2 video: after an append every per-token buffer must have one entry per "
               "token. A buffer that did not grow with the others is invisible to the render's "
               "SHAPE — the clip comes out the right size and simply describes the wrong "
               "tokens.");

      // AND IT LANDED ON THE LAST FRAME. Everything above proves the sequence
      // grew and stayed self-consistent; none of it can see WHERE the appended
      // tokens sit in time, and that is the whole content of this arm. MEASURED:
      // mutation M10 changed `frame_idx` from `frames - 1` to `0` and the suite
      // stayed GREEN — both renders still differed from the no-op control and
      // from each other, and the token count was identical, because a keyframe
      // pinned to the FIRST frame appends exactly as many tokens as one pinned
      // to the last.
      //
      // The expectation is recomputed from `frames` and `fps`, NOT read back
      // from the `frame_idx` argument above, so the two are independent
      // expressions and a mutation of the argument alone moves one and not the
      // other. `Ltx2ConditionVideoByKeyframe` offsets the item's temporal
      // coordinates by `frame_idx` in integer PIXEL space and then divides the
      // temporal axis by fps (keyframe_cond.py:52-59), so the first appended
      // token's temporal START is `frame_idx / fps`. Positions are
      // [pos_dims, tokens, 2] concatenated PER DIMENSION, so the temporal axis
      // is dimension 0 and this item's first token sits at
      // `before_last_frame * 2` — the count at the moment of ITS append, which
      // is the target grid only when nothing appended before it.
      const double want_t0 = static_cast<double>(static_cast<float>(
          static_cast<double>(frames - 1) / fps));
      const double got_t0 = video.positions[static_cast<size_t>(before_last_frame * 2)];
      VT_CHECK(std::abs(got_t0 - want_t0) <= 1e-5 * std::max(1.0, std::abs(want_t0)),
               "ltx2 video: the last-frame keyframe's appended tokens must carry the temporal "
               "position of pixel frame `frames - 1` (" +
                   std::to_string(want_t0) + "), but the first appended token starts at " +
                   std::to_string(got_t0) +
                   ". A keyframe that appends the right number of tokens at the wrong TIME "
                   "renders a clip of the right length that pins the image to the wrong end");
    }

    // ── GENERATED KEYFRAME SLOTS (#986) ─────────────────────────────────────
    //
    // LAST among the appending items, and the order is upstream's: DFR builds
    // `combined_image_conditionings` first and then `append`s the slot item
    // (dfr_pipeline.py:320-330, :353-365). Items are applied in list order and
    // each appends to the END, so putting the slots last is what makes their
    // recorded `first_token` describe the tokens they actually occupy. A slot
    // item applied BEFORE a supplied keyframe would record a layout that the
    // keyframe's own append then pushes nothing off — the layout would still be
    // correct — but the reverse ordering makes the trailing-token assumption
    // look true, and the whole point of the layout is that it is not.
    //
    // SEEDED FROM THE PREVIOUS PHASE on every phase but the first. Upstream's
    // stage 2 passes `initial_keyframes=upsampled_slot_keyframes` (:364), which
    // are stage 1's own denoised slots run through the SPATIAL latent upsampler
    // (:348) — the same operator the video latent takes. Dropping the seed would
    // regenerate every slot from noise at full resolution and lose the base
    // stage's composition, which renders a clip whose keyframes simply disagree
    // with the video around them.
    if (!slot_positions.empty()) {
      // AGAIN THE COUNT AT THE APPEND, not the target grid, and for the reason
      // the last-frame arm states: two supplied keyframes can already stand in
      // front of this item, so `target_tokens` is the length before the FIRST
      // append and describes nothing here (#1219).
      const int64_t before_slots = video.tokens;
      Ltx2LatentState state = ToLatentState(video, /*pos_dims=*/3);
      Ltx2ConditionVideoByGeneratedKeyframeSlots(
          &state, vshape, /*patch_size=*/1, factors, fps, slot_positions,
          slot_keyframes.frames > 0 ? &slot_keyframes : nullptr);
      slot_layout = state.generated_keyframe_layout;
      FromLatentState(state, &video);

      VT_CHECK(video.tokens ==
                   before_slots + static_cast<int64_t>(slot_positions.size()) *
                                      slot_layout.tokens_per_keyframe,
               "ltx2 video: the generated keyframe slots must APPEND one latent frame of tokens "
               "per slot (keyframe_slots.py:83-84, :136-140), and the sequence did not grow by "
               "that many");
      // ...AND THE RECORDED LAYOUT DESCRIBES THOSE TOKENS. `first_token` is the
      // state's own pre-append count (keyframe_slots.py:143-147), and the marked
      // walk below starts there; a layout that named some other token would
      // count the wrong window and still report a healthy total.
      VT_CHECK(slot_layout.first_token == before_slots,
               "ltx2 video: the generated keyframe layout must start at the token the slots were "
               "appended at (keyframe_slots.py:143-147), and it names " +
                   std::to_string(slot_layout.first_token) + " against " +
                   std::to_string(before_slots));
      // THE MARKER REACHED THE NEW TOKENS. This is the one thing the slot arm
      // buys over an ordinary append, it is what `marked=true` exists for
      // (keyframe_slots.py:121), and nothing about the render's shape, its token
      // count or its frame count can see whether it happened. A slot appended
      // with `marked=false` costs the same tokens, renders the same size, and
      // silently omits the trained embedding — which is precisely the polarity
      // `ltx2_conditioning.h` warns about for the first-frame mask.
      VT_CHECK(static_cast<int64_t>(video.keyframes_mask.size()) == video.tokens,
               "ltx2 video: the keyframes mask must cover every token after the slot append");
      int64_t marked = 0;
      for (int64_t t = slot_layout.first_token; t < video.tokens; ++t) {
        if (video.keyframes_mask[static_cast<size_t>(t)] != 0.0F) ++marked;
      }
      // Read off the state the loop will actually run over, not off the request.
      // A trace that restated the request would report a healthy slot count on a
      // build that appended nothing — the failure mode `audio_frozen` already
      // paid for on this struct.
      im.trace.slot_positions = slot_positions;
      im.trace.slot_marked_tokens = marked;
      VT_CHECK(marked == slot_layout.num_tokens(),
               "ltx2 video: every generated keyframe slot token must be MARKED in "
               "`keyframes_mask` (`extend_keyframes_mask(..., marked=True)`, "
               "keyframe_slots.py:121) — upstream's only marked call site, and the whole reason "
               "a slot differs from a supplied keyframe. " +
                   std::to_string(marked) + " of " + std::to_string(slot_layout.num_tokens()) +
                   " are marked. An unmarked slot costs the same tokens and renders the same "
                   "size while omitting the trained embedding.");
    }

    // The noiser draws VIDEO first, AUDIO second, from one generator
    // (blocks.py:554-563 builds the video state before the audio one; :576-580,
    // which this used to cite, is the TEARDOWN and proves nothing about order).
    // The length the DiT will actually run over on this phase, recorded BEFORE
    // the trim and after every conditioning item, so the last phase's value is
    // the one a reader sees. Written inside the loop for the same reason
    // `image_tokens` is: the rest of the trace is filled before denoise and
    // cannot observe anything that happens in here.
    im.trace.video_tokens = video.tokens;

    const float noise_scale = static_cast<float>(phase.noise_scale);
    ApplyGaussianNoise(video, state_noise.Draw(static_cast<int64_t>(video.latent.size())),
                       noise_scale);
    ApplyGaussianNoise(audio, state_noise.Draw(static_cast<int64_t>(audio.latent.size())),
                       noise_scale);

    // The audio arm of the trace (#922), read AFTER the noiser and off the mask
    // the loop will actually use.
    //
    // BOTH OF THOSE ARE THE POINT, and neither was true when this block sat
    // above the noiser and reported the REQUEST's `audio_frozen` flag. MEASURED:
    // with the trace there, a mutation that never zeroed the denoise mask —
    // i.e. that noised and denoised the caller's take instead of holding it —
    // left this suite at 3 cases / 51 assertions / exit 0. The digest was taken
    // before the noise it was supposed to detect, and the flag restated the
    // request rather than observing the state. Reading the mask makes the
    // instrument answer the question the freeze claim actually makes.
    im.trace.audio_frozen =
        !audio.mask.empty() &&
        std::all_of(audio.mask.begin(), audio.mask.end(), [](float m) { return m == 0.0F; });
    im.trace.audio_latent_digest = DigestF32(audio.latent);
    im.trace.audio_latent_absmax = AbsMax(audio.latent);

    // ── the schedule ────────────────────────────────────────────────────────
    std::vector<float> sigmas = phase.sigmas;
    if (sigmas.empty()) {
      int64_t steps = gen.steps > 0 ? gen.steps : recipe.num_inference_steps;
      if (steps < 1) Fail("num_inference_steps resolved to " + std::to_string(steps));
      // `target_tokens`, NOT `video.tokens`, and this line sits AFTER the
      // conditioning block so the distinction is live. Upstream's shift comes
      // from `tokens = math.prod(latent.shape[2:])` (schedulers.py:32) — the
      // UNPATCHIFIED target latent, which by construction cannot see appended
      // tokens — and every pipeline computes its sigmas before a state exists at
      // all (ti2vid_one_stage.py:207 passes no latent; distilled.py:200-201 uses
      // frozen constants). Reading the grown count here would re-shift the whole
      // trajectory the moment a keyframe was supplied, which no shape check and
      // no frame count can see.
      // ONE local feeds both the schedule and the trace, deliberately. If the
      // reported number were written independently of the number passed, a
      // change to the argument alone would leave the trace still reporting the
      // target and the gate still green — the instrument would be describing a
      // build that no longer exists. Bound here so the ordinary mutation moves
      // both. (It does not defend against an edit to the call argument only;
      // nothing local can, and that residual is recorded in the row's spec.)
      //
      // WHICH anchor, though, is the phase's to say. `LTX2Scheduler.execute`
      // takes an OPTIONAL latent, and upstream selects between the target grid
      // and the fixed `default_number_of_tokens` = 4096 by passing one or not
      // (schedulers.py:31). Six of upstream's seven call sites pass none;
      // `ti2vid_two_stages_hq.py:267` is the one that does. See
      // `Ltx2PhaseScheduleTokens`, whose default is this engine's long-standing
      // `target_tokens` and whose divergence from upstream's majority is #1150.
      //
      // Resolved to a CONCRETE count rather than passing 0 for "take the
      // default". `Ltx2SigmaSchedule` treats the two identically
      // (ltx2_pipeline.cpp's `tokens > 0 ? tokens : default`), and the concrete
      // form keeps the one-local property below true for BOTH branches: the
      // trace then reports 4096 rather than a sentinel, so a gate can assert the
      // anchor by equality instead of by absence.
      const int64_t schedule_tokens =
          phase.schedule_tokens == Ltx2PhaseScheduleTokens::kSchedulerDefault
              ? Ltx2SchedulerParams{}.default_number_of_tokens
              : target_tokens;
      sigmas = Ltx2SigmaSchedule(steps, schedule_tokens);
      im.trace.schedule_tokens = schedule_tokens;
    } else if (gen.steps > 0 && !recipe.allow_request_sigmas) {
      // `fixed_num_inference_steps` (ltx2_recipes.py:53-87): a distilled recipe's
      // schedule is trained INTO the model, so honouring a step override would
      // sample a trajectory the weights were never distilled for.
      Fail("this recipe fixes its own distilled schedule (" +
           std::to_string(static_cast<int64_t>(phase.sigmas.size()) - 1) + " steps for phase '" +
           phase.name + "'), so a `steps` override is refused rather than applied");
    }

    // This phase's two guiders, resolved once. `GuidedDenoiser` is constructed
    // per stage upstream and holds its guiders for the whole loop
    // (ti2vid_one_stage.py:221-226, ti2vid_two_stages_hq.py:271-281), so
    // resolving them per step would let a request override change meaning
    // halfway down a schedule.
    const Ltx2MultiModalGuiderParams& video_guidance =
        phase_guidance[static_cast<size_t>(phase_index)].video;
    const Ltx2MultiModalGuiderParams& audio_guidance =
        phase_guidance[static_cast<size_t>(phase_index)].audio;
    if (phase_index == 0) {
      im.trace.video_guidance_cfg_scale = video_guidance.cfg_scale;
      im.trace.video_guidance_stg_scale = video_guidance.stg_scale;
      im.trace.video_guidance_rescale_scale = video_guidance.rescale_scale;
      im.trace.video_guidance_modality_scale = video_guidance.modality_scale;
    }

    // `_last_denoised_video` / `_last_denoised_audio` (denoisers.py:274-275):
    // per DENOISER, so per phase, and empty until the first step fills them. A
    // skipped step reuses them instead of running a forward.
    std::vector<float> last_denoised_video;
    std::vector<float> last_denoised_audio;
    // Phase 0's FIRST evaluation is what `RecordFirstGuidedStep` describes. On
    // the first-order arm that is step 0, which is what this was before the
    // res_2s loop existed; on the res_2s arm it is the first of that step's TWO
    // evaluations (samplers.py:301), because the second one runs over a midpoint
    // state and at a substep sigma and would describe a different call.
    // AND THE SECOND EVALUATION IS RECORDED SEPARATELY, which is why this is a
    // counter rather than a bool. The res_2s substep runs over `x_mid`, a state
    // that never becomes the stream's own latent (samplers.py:369-378), so the
    // x0 conversion there is the one place in this file where "the latent" and
    // "the latent this evaluation was handed" are different tensors. MEASURED:
    // with the conversion reading `video.latent` instead, the whole
    // `test_ltx2_video` suite stayed GREEN at 74 cases and 2234 assertions —
    // the clip, the counts, the eval sigmas and the bong count are all blind to
    // it, because the loop's own arithmetic is gated with a FIXTURE denoiser and
    // the engine's conversion is not in that loop.
    int64_t phase_evaluation_index = 0;

    // ── ONE EVALUATION, SHARED BY EVERY SAMPLER ─────────────────────────────
    //
    // Upstream's samplers all take a `Denoiser` callable and never reach for a
    // model (samplers.py:213-214, :45), which is why the loops differ only in
    // how many times, at which sigmas and at which step indices they call it.
    // This lambda is that callable, and BOTH arms below go through it: the
    // first-order loop calls it once per step, `Ltx2Res2sDenoisingLoop` calls it
    // twice per step plus once at the end.
    //
    // AND IT IS THE GUIDED DENOISER, on both arms. Upstream's HQ stage 1 hands
    // `res2s_audio_video_denoising_loop` a `GuidedDenoiser`
    // (ti2vid_two_stages_hq.py:271-281, :292) exactly as the one-stage pipeline
    // hands its Euler loop one (ti2vid_one_stage.py:221-226), so the sampler
    // decides HOW MANY denoiser calls happen and the denoiser decides how many
    // forwards each call is. Routing res_2s around `Ltx2GuidedDenoise` would
    // make the HQ preset the only unguided video arm in the tree — a plausible
    // clip at cfg 1.0 where the preset was tuned at 3.0 — and the evaluation
    // count, which is what this row's gate reads, would not move by one.
    //
    // Hoisted rather than duplicated because a second forward path written by
    // hand would be a second place to forget the keyframe marker, the frozen
    // scalar sigma or the device/host split — and every one of those omissions
    // renders a finished clip. It also makes `dit_evaluations` a single
    // increment that no arm can bypass.
    //
    // It takes the latent as an ARGUMENT rather than reading `video.latent`,
    // because the res_2s loop's second evaluation runs over a MIDPOINT state
    // that never becomes the stream's own latent (samplers.py:369-378).
    //
    // `sigma` is a `double` on the way in and narrows here. That narrowing is
    // upstream's own boundary rather than a shortcut: `Modality.sigma` reaches
    // the DiT as a tensor of the model's dtype, and this port's
    // `Ltx2ModalityInput::sigma` is a `const float*`. The res_2s substep sigma
    // is float64 up to this line (samplers.py:315, :384) and float32 after it.
    //
    // `step_index` IS THE DENOISER'S OWN ARGUMENT, not the sampler's loop
    // counter. Upstream's `Denoiser` signature is
    // `denoiser(transformer, video_state, audio_state, sigmas, step_index)`, and
    // the res_2s loop passes THREE different values for it: `step_idx` at the
    // first evaluation (samplers.py:301), a literal `0` at the substep
    // evaluation beside a one-element schedule (samplers.py:384-385), and
    // `n_full_steps` at the terminal one (samplers.py:437). It is read by
    // `should_skip_step` (`step % (skip_step + 1) != 0`, guiders.py:287-291), so
    // the substep evaluation is NEVER skipped whatever `skip_step` is. That is
    // inert on the HQ preset, whose `skip_step` is 0 (constants.py:104, :112),
    // and it is NOT inert for a request that sets `video_skip_step`. Passing the
    // loop counter here instead would skip half of a res_2s step's evaluations
    // on such a request and render at the first-order method's cost with the
    // second-order sampler's schedule.
    const auto Evaluate = [&](const std::vector<float>& v_latent,
                              const std::vector<float>& a_latent, double sigma_hp,
                              int64_t step_index, std::vector<float>& v_denoised,
                              std::vector<float>& a_denoised) {
      // W0 repair (#1010, second fresh review): THE SUB-SCOPE THAT TIES THE
      // `denoise` NAME TO THE WORK BENEATH IT.
      //
      // The first review's M4 showed that existence plus a sum cannot see a leaf
      // whose name sits on somebody else's seconds. The repair asserted
      // CONTAINMENT — and asserted it for `decode.audio` alone, because that was
      // the only phase with sub-scopes. The second review then ran the same
      // mutation one level over, on the phase that carries this render: close
      // `denoise` after the first sampler step and open `phase.finish` there. No
      // overlap, no nesting, the sum untouched, 99.94% accounted — and 82% of the
      // denoise re-labelled with BOTH gates green.
      //
      // This is the anchor that closes it. Every sampler arm reaches `Evaluate`,
      // and `Evaluate` is the denoiser evaluation itself, so one nested leaf per
      // evaluation says where the sampler actually spent its time. A `denoise`
      // leaf that stops short of the loop no longer contains its own steps.
      // NESTED, so the sum does not move: the marking is automatic, because a
      // leaf opened while `denoise` is live is excluded from `Sum`.
      const ::vllm::multimodal::phase::Scope step_phase("denoise.step");
      const float sigma = static_cast<float>(sigma_hp);
      const std::vector<float> v_timesteps = TimestepsFromMask(video, sigma);
      const std::vector<float> a_timesteps = TimestepsFromMask(audio, sigma);
      // The SECOND half of upstream's `frozen` on the VIDEO side
      // (retake.py:274, utils/types.py:104-106), by the same argument the audio
      // side gives below: the zeroed denoise mask carries the per-token half,
      // and this scalar is a separate DiT input the mask cannot reach.
      const float sigma_row = (wants_retake && retake_plan.video_frozen) ? 0.0F : sigma;
      // Observed rather than asserted in prose, for the reason `audio_sigma_max`
      // records below its own field: on the audio side the same claim survived a
      // mutation while it was only a comment.
      im.trace.video_sigma_max =
          std::max(im.trace.video_sigma_max, static_cast<double>(sigma_row));

      Ltx2ModalityInput vin;
      vin.batch = 1;
      vin.tokens = video.tokens;
      vin.context_tokens = context_tokens;
      vin.latent = v_latent.data();
      vin.timesteps = v_timesteps.data();
      vin.sigma = &sigma_row;
      vin.positions = video.positions.data();
      vin.context = video_context;
      // transformer_args.py:269 through modality.py:63. Handed over only when the
      // model HAS the parameter: a mask on a model without one is refused by the
      // forward, and upstream's own `supports_keyframes_abs_pos_embedding`
      // (model.py:166-173) is exactly this condition. The mask itself is built
      // unconditionally above, because it is data about the latent.
      //
      // THE EMPTINESS IS CHECKED, and that is not defensive noise. Making the
      // mask conditional — on `wants_image`, on a keyframe, on anything — is the
      // one defect this module invites, and it is INVISIBLE to every output
      // check: the render stays finite, the right shape, the right token count,
      // and simply omits a trained term. Without this line a conditional mask
      // reaches the DiT as `data()` on an empty vector, which is a null pointer
      // and therefore upstream's legal "no token is marked" — a silent drop
      // dressed as a supported path. MEASURED: with the mask made conditional
      // and this check absent, all five LTX-2.5 suites stayed GREEN.
      if (im.dit.params.use_keyframes_abs_pos_embedding) {
        VT_CHECK(static_cast<int64_t>(video.keyframes_mask.size()) == video.tokens,
                 "ltx2 video: this DiT carries keyframes_abs_pos_embedding, so every forward owes "
                 "the marker `_first_frame_keyframes_mask` builds (ltx_core/tools.py:184-196) — "
                 "one value per video token, populated on EVERY generation whether or not a "
                 "keyframe was supplied. Handing the forward no marker would render without a "
                 "trained term and look exactly like a working render.");
        vin.keyframes_mask = video.keyframes_mask.data();
      }
      // AND THE HANDOVER IS CHECKED SEPARATELY FROM THE CONSTRUCTION, because the
      // check above cannot see the handover. It reads `video.keyframes_mask` — the
      // VECTOR — so it fires when the mask is built conditionally and stays silent
      // when the ASSIGNMENT is. MEASURED: with the vector left unconditional and
      // this assignment written `if (wants_image) vin.keyframes_mask = ...`, all
      // five LTX-2.5 suites stayed GREEN while the rendered pixels moved — frame 0
      // went from a flat 127 to a flat 130 — so the drop was real and nothing in
      // the tree named it. Two sites, one invariant, and the earlier guard covered
      // only one of them.
      //
      // `vin.keyframes_mask` is the field `Ltx2DitForward` reads, so it is the only
      // fact that decides whether the trained term is applied. The condition stays
      // on the flag rather than becoming a bare `!= nullptr`: a DiT that does NOT
      // carry the parameter must reach the forward with a null marker, which is
      // upstream's `keyframes_mask is None` exit and is itself gated at
      // `ltx2_dit.cpp`'s `m.keyframes_mask == nullptr || keyframes_embedding !=
      // nullptr`. Handing that model a marker would be a refusal, not a fix.
      VT_CHECK(!im.dit.params.use_keyframes_abs_pos_embedding || vin.keyframes_mask != nullptr,
               "ltx2 video: this DiT carries keyframes_abs_pos_embedding, so the forward owes the "
               "marker on EVERY step — and this forward was handed none. "
               "`_first_frame_keyframes_mask` (ltx_core/tools.py:184-196) is built on the same "
               "line as the state, unconditionally, whether or not a keyframe or an image was "
               "supplied. A marker that is BUILT and then not HANDED OVER renders without a "
               "trained term and looks exactly like a working render.");

      Ltx2ModalityInput ain;
      ain.batch = 1;
      ain.tokens = audio.tokens;
      ain.context_tokens = context_tokens;
      ain.latent = a_latent.data();
      ain.timesteps = a_timesteps.data();
      // The SECOND half of upstream's `frozen` (utils/types.py:104-106): the
      // per-modality scalar sigma is forced to 0, "not only per-token
      // timesteps". The zeroed denoise mask above already carries the per-token
      // half through `TimestepsFromMask`, and this scalar is a separate input to
      // the DiT that the mask cannot reach. Leaving it at the schedule's sigma
      // would tell the model the audio it is conditioning on is noisy when it is
      // the caller's own clean take — a wrong conditioning signal that still
      // renders a finished clip.
      const float audio_sigma_row = audio_frozen ? 0.0F : sigma_row;
      ain.sigma = &audio_sigma_row;
      // Observed, not asserted in prose: see `audio_sigma_max` in the header for
      // the mutation that survived while this was only a comment.
      im.trace.audio_sigma_max =
          std::max(im.trace.audio_sigma_max, static_cast<double>(audio_sigma_row));
      ain.positions = audio.positions.data();
      ain.context = audio_context;

      // ── the X0 MODEL (model.py:590-604), and the guided denoiser ──────────
      //
      // `DiffusionStage` never hands the loop the raw velocity model: it hands
      // `X0Model(builder.build(...))` (utils/blocks.py:480-482, the forward it
      // wraps at ltx-core model/transformer/model.py:590-604). So `to_denoised`
      // belongs HERE, inside the wrapper, applied to EVERY pass on its way out of
      // the forward — and the guider downstream combines already-denoised
      // tensors. Converting once after the guider instead is a DIFFERENT function
      // wherever `rescale_scale != 0` (guiders.py:268-271), which is 0.7 on every
      // video row of the params table. That defect shipped on the audio arm of
      // this tree and is #1039.
      //
      // One graph, two residencies. On the CPU this is the L2 parity forward in
      // its declared f32; on an accelerator it is the phase-L8 device-resident
      // forward over the bf16 the DiT was STAGED at, and the two agree on
      // everything but where the bytes live and how wide they are — INCLUDING
      // `perturbations`, for which the device forward took no argument until
      // #1092's second half and which is why the whole perturbed arm was refused
      // there.
      //
      // `p` IS PASSED ON BOTH BRANCHES, and that symmetry is the point: an
      // argument list that carries the perturbation on one residency and drops it
      // on the other renders, with the STG and modality terms at exactly zero and
      // nothing in the frames, the shapes or the counts to show for it. No test on
      // a box without an accelerator enters the first branch, because
      // `Ltx2VideoEngine::Load` refuses `device != 0` where no accelerator backend
      // is registered — so that half is link B of
      // .agents/specs/ltx25-guided-video.md §12.8, listed under its `## Owed` and
      // owed a leased run rather than left implied.
      const Ltx2X0Model x0_model = [&](const Ltx2ModalityInput* v, const Ltx2ModalityInput* a,
                                       const Ltx2DitPerturbation* p) {
        // W0-live (#1413): THE tick, and it fires BEFORE the forward rather than
        // after it. At #1375's measured ~162 s per forward on the 21.004 B model
        // this line is the only output the process produces for minutes at a
        // time, so it has to name the forward that is IN FLIGHT — otherwise a run
        // killed inside forward 37 says 36 and a reader has to guess whether the
        // missing one ran. The cost of that choice: the last forward of a
        // completed render has no line of its own, and its duration is inside the
        // `- denoise` boundary line.
        //
        // NO DENOMINATOR ON THE FORWARD COUNTER, deliberately. 60 = 30 x 2 is
        // true of the config #1375 measured and is not structural here: the
        // sampler decides how many denoiser calls happen and `Ltx2GuidedDenoise`
        // decides how many forwards each call is (one to four — cond, uncond,
        // ptb, mod, denoisers.py:100-137). The STEP fraction is exact, because
        // `sigmas` is this recipe phase's own schedule; note that the res_2s
        // substep passes a literal 0 and its terminal evaluation passes
        // `n_full_steps`, so on that arm the step number is the DENOISER's index
        // and not a monotone loop counter. The forward counter is the monotone
        // one, which is why the progress claim rests on it.
        //
        // GUARDED AT THE CALL SITE, not only inside `Tick`. `Tick` returns on
        // its first line when the lane is off, but `detail` is already built by
        // then: one `std::string` from a literal, three `std::to_string`s and
        // four concatenations, per forward. Negligible against 162 s and not
        // zero, and the spec claims `VLLM_RENDER_PROGRESS=0` costs one `getenv`
        // per process — which is true only with this `if` here.
        if (::vllm::multimodal::phase::ProgressEnabled()) {
          ::vllm::multimodal::phase::Tick(
              "dit forward", im.trace.dit_forwards + 1,
              "phase " + std::to_string(phase_index) + " step " +
                  std::to_string(step_index + 1) + "/" +
                  std::to_string(static_cast<int64_t>(sigmas.size()) - 1));
        }
        const Ltx2DitOutputs velocity =
            im.on_device ? Ltx2DitForwardDevice(*im.queue, im.dit.params, im.dit.weights, v, a,
                                                im.compute_dtype, /*cache=*/nullptr, p)
                         : Ltx2DitForward(im.device, im.dit.params, im.dit.weights, v, a,
                                          im.compute_dtype, /*cache=*/nullptr, p);
        // EVERY ACTUAL DiT FORWARD IS COUNTED HERE, and that is a different
        // number from `dit_evaluations` one level up. One denoiser evaluation is
        // one to four forwards (cond, uncond, ptb, mod — denoisers.py:100-137),
        // so the two counters answer two questions that no output can: WHICH
        // SAMPLER ran, and WHETHER THE ARM WAS GUIDED. An unguided HQ render
        // keeps `dit_evaluations` at 2n+1 and drops this one from 3(2n+1) to
        // 2n+1, and nothing else about the clip changes.
        im.trace.dit_forwards += 1;
        Ltx2X0Outputs out;
        out.video_velocity = velocity.video;
        out.audio_velocity = velocity.audio;
        // The PER-TOKEN timesteps, not the schedule scalar: a conditioned token
        // sits at timestep 0 and using the scalar there re-noises it.
        //
        // AND THE LATENT IS THE ONE THIS EVALUATION WAS HANDED, not the stream's
        // own. They are the same tensor on the first-order arm and on the res_2s
        // first evaluation, and they are NOT the same on the res_2s substep
        // evaluation, which runs over `x_mid` (samplers.py:369-378). Reading
        // `video.latent` here would convert the substep's velocity against the
        // wrong sample and still return a finite, correctly shaped prediction.
        out.video = ToDenoised(v_latent, velocity.video, v_timesteps, video.tokens, video.width);
        out.audio = ToDenoised(a_latent, velocity.audio, a_timesteps, audio.tokens, audio.width);
        return out;
      };

      Ltx2GuidedDenoiseInputs denoise_in;
      denoise_in.video = &vin;
      denoise_in.audio = &ain;
      denoise_in.video_negative_context = negative_video_context;
      denoise_in.audio_negative_context = negative_audio_context;
      denoise_in.video_guider = video_guidance;
      denoise_in.audio_guider = audio_guidance;
      denoise_in.num_blocks = im.dit.params.num_layers;
      denoise_in.step_index = step_index;
      denoise_in.last_denoised_video = &last_denoised_video;
      denoise_in.last_denoised_audio = &last_denoised_audio;
      const Ltx2GuidedDenoiseResult guided = Ltx2GuidedDenoise(x0_model, denoise_in);

      // THE ONE PLACE A DENOISER EVALUATION IS COUNTED. Every sampler reaches
      // it, so a build that ran the wrong number of them cannot report the right
      // count. This is the only observable that separates the res_2s sampler
      // from the first-order one — the clip, its shape, its frame count, its
      // sample rate and its file size are identical between them — which is why
      // it is a counter rather than a comment. `dit_forwards` inside the x0
      // model above is the other half: this one counts CALLS, that one counts
      // FORWARDS, and only the second moves when guidance is dropped.
      im.trace.dit_evaluations += 1;
      // W0: one host/device sample per DiT forward. The scope boundaries alone
      // would report the peak a MINUTES-long denoise reached only at its ends,
      // and W6's attribution of the ~59 GiB needs the interior.
      ::vllm::multimodal::phase::SampleNow();

      // `last_denoised_*` keeps what the GUIDER returned, before the
      // post-process, because that is what `_last_denoised_video` holds
      // (denoisers.py:299-300) and what a skipped step reuses.
      last_denoised_video = guided.video_denoised;
      last_denoised_audio = guided.audio_denoised;

      if (phase_index == 0 && phase_evaluation_index == 0) {
        // `stepper_input` is the POST-PROCESSED prediction, which is what both
        // samplers hand their stepper: the first-order loop through
        // `_step_state` (samplers.py:35) and the res_2s loop at :305. Computed
        // here rather than taken from the caller so the res_2s arm, whose
        // post-process runs inside the sampler at f64, records the same quantity
        // the Euler arm does.
        RecordFirstGuidedStep(&im.trace, guided, v_latent, v_timesteps,
                              static_cast<double>(sigma),
                              PostProcessLatent<float>(guided.video_denoised, video));
      }
      // THE SUBSTEP EVALUATION, whose x0 conversion has no other observable.
      // Recorded on the res_2s arm alone, because on a first-order arm the
      // second evaluation is just step 1 and `video_first_*` already describes
      // the shape. See `res2s_substep_*` in ltx2_video.h.
      if (phase_index == 0 && phase_evaluation_index == 1 &&
          phase.stepper == Ltx2StepperKind::kRes2s) {
        const size_t cond = static_cast<size_t>(Ltx2DenoisePass::kCond);
        im.trace.res2s_substep_latent = v_latent;
        im.trace.res2s_substep_timesteps = v_timesteps;
        im.trace.res2s_substep_cond = guided.video_pass[cond];
        im.trace.res2s_substep_cond_velocity = guided.video_pass_velocity[cond];
        im.trace.res2s_substep_sigma = static_cast<double>(sigma);
      }
      phase_evaluation_index += 1;

      // RAW, not post-processed. `post_process_latent` belongs to the SAMPLER
      // upstream, not to the denoiser: the first-order loop applies it inside
      // `_step_state` (samplers.py:35) and the res_2s loop applies it at four
      // separate points (:305, :390, :203, :441), one of which is after an SDE
      // injection rather than after an evaluation. Folding it in here would put
      // it in three of those four places and silently drop the fourth.
      v_denoised = guided.video_denoised;
      a_denoised = guided.audio_denoised;
    };

    // ── the denoise loop ────────────────────────────────────────────────────
    // The ancestral arm's loop generator is seeded from the pipeline seed plus
    // the recipe's own offset (distilled.py:69-73, :178-184) — a separate stream
    // from the state noise, so its first draw is not the initial latent's.
    SplitMixGaussian loop_noise(seed + static_cast<uint64_t>(phase.noise_seed_offset));
    const int64_t sigma_count = static_cast<int64_t>(sigmas.size());

    phase_prepare.Close();
    // W0: THE phase #1024 was read as, and the one this instrument exists to
    // separate from the rest. It covers both sampler arms, because from the
    // outside they are the same wait.
    ::vllm::multimodal::phase::Scope denoise_phase("denoise");
    if (phase.stepper == Ltx2StepperKind::kRes2s) {
      // ── the res_2s second-order sampler (samplers.py:208-447) ─────────────
      //
      // Row LTX25-RES2S-LOOP, issue #921. `TI2VidTwoStagesHQPipeline` passes
      // `loop=res2s_audio_video_denoising_loop` to BOTH of its stages
      // (ti2vid_two_stages_hq.py:292, :335), and this is that loop.
      //
      // THE PARAMETERS ARE THE LOOP'S OWN DEFAULTS, DELIBERATELY.
      // `DiffusionStage.__call__` hands the loop six keyword arguments and no
      // others (utils/blocks.py:566-573), so nothing on the HQ path overrides
      // eta, bongmath, the iteration cap, the noise function or the seeds.
      // Passing anything else here would be this port inventing a knob.
      //
      // THE SEEDS ARE CONSTANTS AND NOT `seed`. `noise_seed` defaults to -1
      // (samplers.py:215) and the substep stream to -1 + 10000
      // (samplers.py:265-266), so the res_2s SDE injections do not depend on the
      // request's seed at all — the initial latent still does, through the
      // noiser. The ancestral arm one branch up does the opposite. Mirrored
      // rather than made consistent, because consistency here would be a
      // divergence.
      SplitMixGaussian res2s_step_noise(static_cast<uint64_t>(kLtx2Res2sNoiseSeed));
      SplitMixGaussian res2s_substep_noise(
          static_cast<uint64_t>(kLtx2Res2sNoiseSeed + kLtx2Res2sNoiseSeedSubstepOffset));

      Ltx2Res2sHooks hooks;
      hooks.denoise = Evaluate;
      hooks.post_process = [&](std::vector<double> x, bool is_video) {
        return PostProcessLatent<double>(x, is_video ? video : audio);
      };
      // `_get_new_noise` (samplers.py:164-170): draw, then normalize. The DRAW
      // is this port's `SplitMixGaussian` rather than upstream's seeded
      // `torch.randn`, so the stream differs — as it already does on the
      // shipped ancestral arm — and only the normalization is mirrored. Which
      // NOISE FUNCTION each loop uses is mirrored too, and the two loops do not
      // agree: the ancestral one defaults to the un-normalized
      // `_get_plain_noise` (samplers.py:574).
      hooks.new_noise = [&](int64_t count, bool /*is_video*/, bool substep) {
        SplitMixGaussian& stream = substep ? res2s_substep_noise : res2s_step_noise;
        const std::vector<float> raw = stream.Draw(count);
        std::vector<double> noise =
            Ltx2Res2sNormalizeNoise(std::vector<double>(raw.begin(), raw.end()));
        // OBSERVED, not asserted in prose. Whether this hook normalizes is
        // invisible in the rendered clip, the token count and the evaluation
        // count alike, and a build that returned `raw` here left the whole
        // end-to-end suite green. See `res2s_noise_moment_error`.
        double mean = 0.0;
        for (const double v : noise) mean += v;
        mean /= static_cast<double>(noise.size());
        double sq = 0.0;
        for (const double v : noise) sq += (v - mean) * (v - mean);
        const double sd = std::sqrt(sq / static_cast<double>(noise.size() - 1));
        im.trace.res2s_noise_moment_error = std::max(
            im.trace.res2s_noise_moment_error, std::max(std::fabs(mean), std::fabs(sd - 1.0)));
        return noise;
      };

      Ltx2Res2sModality res2s_video{video.latent, true};
      Ltx2Res2sModality res2s_audio{audio.latent, true};
      // TWO INDEPENDENT COUNTERS, and the check below is only worth running
      // because they are independent. `stats.evaluations` is the LOOP's own
      // count; `im.trace.dit_evaluations` is incremented inside `Evaluate`, i.e.
      // by the ENGINE, once per call the loop actually made. This delta is what
      // makes the comparison an observation rather than an identity: the
      // previous form of this check compared `stats.evaluations` against
      // `stats.full_steps`, both fields of the same struct, and `2n + 1 > n`
      // holds for every n >= 1, so it could not fail for any build.
      const int64_t evaluations_before = im.trace.dit_evaluations;
      const Ltx2Res2sLoopStats stats =
          Ltx2Res2sDenoisingLoop(sigmas, res2s_video, res2s_audio, hooks);
      video.latent = std::move(res2s_video.latent);
      audio.latent = std::move(res2s_audio.latent);
      const int64_t engine_evaluations = im.trace.dit_evaluations - evaluations_before;
      VT_CHECK(engine_evaluations == stats.evaluations,
               "ltx2 video: the res_2s loop reports " + std::to_string(stats.evaluations) +
                   " denoiser evaluations and the engine counted " +
                   std::to_string(engine_evaluations) + ". The loop counts its own calls and the "
                   "engine counts the ones that reached `Evaluate`, so a disagreement means a "
                   "call was made without reaching the shared evaluation — the one place the "
                   "keyframe marker, the frozen scalar sigma, the guided denoiser and the "
                   "host/device split are all applied.");
      VT_CHECK(engine_evaluations > stats.full_steps,
               "ltx2 video: the res_2s sampler evaluates the denoiser TWICE per step plus once at "
               "a terminal zero sigma (samplers.py:301, :380-386, :437), so the engine cannot "
               "count as many evaluations as the loop has steps. A count at or below the step "
               "count means the second evaluation was skipped, which renders a finished, "
               "correctly sized, plausible clip at half the model evaluations the HQ preset was "
               "tuned for.");
      im.trace.res2s_bong_steps += stats.bong_steps;
    } else {
      for (int64_t step = 0; step + 1 < sigma_count; ++step) {
        const float sigma = sigmas[static_cast<size_t>(step)];
        std::vector<float> v_raw, a_raw;
        // `step` IS the denoiser's `step_index` on this arm — upstream's
        // first-order loop passes its own loop counter straight through
        // (samplers.py:45, :503) — which is what `should_skip_step` reads.
        Evaluate(video.latent, audio.latent, static_cast<double>(sigma), step, v_raw, a_raw);
        // `_step_state` (samplers.py:35) blends before it steps.
        const std::vector<float> v_denoised = PostProcessLatent<float>(v_raw, video);
        const std::vector<float> a_denoised = PostProcessLatent<float>(a_raw, audio);

        const bool terminal = sigmas[static_cast<size_t>(step + 1)] == 0.0F;
        if (phase.stepper == Ltx2StepperKind::kEulerAncestral) {
          if (terminal) {
            // samplers.py:545-547 — the terminal step IS the denoised
            // prediction; taking an ancestral step there would re-noise the
            // finished latent.
            video.latent = v_denoised;
            audio.latent = a_denoised;
            if (phase_index == 0 && step == 0) im.trace.video_first_next_latent = video.latent;
            continue;
          }
          const std::vector<float> v_noise =
              loop_noise.Draw(static_cast<int64_t>(video.latent.size()));
          const std::vector<float> a_noise =
              loop_noise.Draw(static_cast<int64_t>(audio.latent.size()));
          video.latent = PostProcessLatent<float>(
              Ltx2EulerAncestralStep(video.latent.data(), v_denoised.data(), sigmas.data(),
                                     sigma_count, step,
                                     static_cast<int64_t>(video.latent.size()),
                                     phase.stepper_eta, phase.stepper_s_noise, v_noise.data()),
              video);
          audio.latent = PostProcessLatent<float>(
              Ltx2EulerAncestralStep(audio.latent.data(), a_denoised.data(), sigmas.data(),
                                     sigma_count, step,
                                     static_cast<int64_t>(audio.latent.size()),
                                     phase.stepper_eta, phase.stepper_s_noise, a_noise.data()),
              audio);
        } else {
          video.latent = Ltx2EulerStep(video.latent.data(), v_denoised.data(), sigmas.data(),
                                       sigma_count, step,
                                       static_cast<int64_t>(video.latent.size()));
          audio.latent = Ltx2EulerStep(audio.latent.data(), a_denoised.data(), sigmas.data(),
                                       sigma_count, step,
                                       static_cast<int64_t>(audio.latent.size()));
        }
        // What the sampler WROTE, recorded after the step rather than derived
        // from what was recorded before it. It is the only observable that says
        // which tensor the stepper was actually handed: a second `ToDenoised` on
        // the way in leaves every other recorded field untouched.
        if (phase_index == 0 && step == 0) im.trace.video_first_next_latent = video.latent;
      }
    }
    denoise_phase.Close();
    // W0: what a recipe phase does AFTER its sampler — the trim, the slot
    // extraction and the pool drain. Small on this fixture and not obviously
    // small at 22B, which is the reason it is named rather than folded in.
    ::vllm::multimodal::phase::Scope phase_finish("phase.finish");

    // `clear_conditioning` + `unpatchify` (blocks.py:575-580, in that order).
    //
    // The clear used to be an explicit identity because nothing could append.
    // Row LTX25-TOKEN-APPEND (#930) made it real: it truncates `latent`, `clean`
    // and `positions` back to the target grid and restores an all-ones denoise
    // mask (tools.py:101-105).
    //
    // WHY THE GUARD IS HERE AND NOT LEFT IMPLICIT. `Ltx2VideoUnpatchify` takes a
    // BARE POINTER, so it cannot tell a target-length buffer from a longer one —
    // and the appended tokens sit at the tail of a contiguous [tokens, width]
    // buffer, so an un-trimmed state would unpatchify the same head bytes and
    // render pixel-identical frames. That is exactly the shape of defect this
    // project keeps finding: correct output for the wrong reason, with no
    // instrument that can see the difference. The check is what turns "the head
    // happens to be right" into "the buffer IS the target grid", and it is what
    // makes deleting the trim a RED rather than a silent pass.
    //
    // AND IT EXTRACTS THE GENERATED KEYFRAME SLOTS BEFORE IT TRIMS (#986).
    // Upstream's order, `tools.py:97` against `:115`, and the order is the whole
    // content: the slots live OUTSIDE the target grid, so this trim is exactly
    // what destroys them. Extracting afterwards returns a perfectly correct
    // video latent and no slots, with nothing in the shape, the token count or
    // the frame count to show for it.
    {
      Ltx2LatentState finished = ToLatentState(video, /*pos_dims=*/3);
      finished.generated_keyframe_layout = slot_layout;
      Ltx2LatentVolume extracted;
      Ltx2ClearConditioning(&finished, target_tokens, &vshape, /*patch_size=*/1, &extracted);
      FromLatentState(finished, &video);

      if (!slot_positions.empty()) {
        VT_CHECK(extracted.frames == static_cast<int64_t>(slot_positions.size()),
                 "ltx2 video: `clear_conditioning` must return one latent frame per generated "
                 "keyframe slot (ltx_core/tools.py:203-230) and it returned " +
                     std::to_string(extracted.frames) + " for " +
                     std::to_string(slot_positions.size()) + " slots");
        // THE SLOTS CARRY CONTENT. The extraction can be structurally perfect
        // and still read a range the denoise never touched — a slot seeded with
        // zeros, held at denoise mask 0 by a wrong polarity, comes back as
        // exactly the zeros it went in as, in the right shape, at the right
        // count. Upstream pins the slots at `denoise_mask = 1`
        // (keyframe_slots.py:118-119) precisely so the loop fills them, so an
        // all-zero readback means the mask, not the layout, is wrong.
        VT_CHECK(AbsMax(extracted.data) > 0.0,
                 "ltx2 video: every extracted generated keyframe slot is exactly zero. The "
                 "layout located a token range the denoise loop never wrote, which is what a "
                 "slot pinned at denoise mask 0 looks like — upstream sets the slot mask to ONE "
                 "(keyframe_slots.py:118-119) so the loop generates into it, and the clean "
                 "tensor it would otherwise lerp toward is deliberately zeros (:139).");
        im.trace.slot_tokens_extracted = extracted.frames;
        slot_keyframes = extracted;
      }
    }
    VT_CHECK(video.tokens == target_tokens &&
                 static_cast<int64_t>(video.latent.size()) == target_tokens * video.width,
             "ltx2 video: the latent handed to unpatchify must be exactly the target grid. "
             "`clear_conditioning` (ltx_core/tools.py:88-117) is what establishes that after an "
             "appending conditioning item, and `Ltx2VideoUnpatchify` takes a bare pointer that "
             "cannot check it.");
    video_latent_volume = Ltx2VideoUnpatchify(video.latent.data(), vshape, 1);
    video_lc = vshape.channels;
    video_lf = vshape.frames;
    video_lh = vshape.height;
    video_lw = vshape.width;
    audio_latent_volume = Ltx2AudioUnpatchify(audio.latent.data(), ashape);
    audio_lc = ashape.channels;
    audio_lf = ashape.frames;
    audio_lm = ashape.mel_bins;

    // PHASE CHANGE. The denoise loop just left the shared scratch pool holding
    // every activation size class this phase touched, and on an UNCAPPED pool
    // (`device_pool_cap_bytes == 0`, which is GB10 and Thor today) those blocks
    // are never returned to the driver. What comes next allocates DIFFERENT
    // classes — the next phase runs at twice the resolution, and after the last
    // phase the VAE decode runs on the host — so none of them can be reused and
    // all of them are headroom the next stage does not get.
    //
    // On GB10 that headroom is HOST memory: the unified pool is one ~119 GiB
    // arena, and this class of box REBOOTS rather than OOM-killing when the
    // driver runs out (`NVRM ... Out of memory [NV_ERR_NO_MEMORY]`), which it did
    // twice during phase L9b's 320x192/25-frame attempt. MiniMax-H3 drains at
    // exactly this boundary and for exactly this reason
    // (minimax_h3_pipeline.cpp:546-563); the LTX path drained nowhere.
    //
    // Draining costs one free per retained block, once per phase.
    if (im.on_device) {
      // A MEASUREMENT LANE, in the same shape `VT_POOL_BYPASS` and
      // `VT_POOL_EXACT` already take (device_pool.h): it exists so the A/B that
      // establishes what the drain is worth runs on ONE binary, which is what
      // AGENTS.md asks of a measurement. It is never a configuration — the drain
      // is on by default and there is no supported reason to turn it off.
      const char* off = std::getenv("VLLM_LTX2_POOL_DRAIN");
      if (off == nullptr || off[0] != '0') {
        vt::Backend& backend = vt::GetBackend(im.device.type);
        // `ActivePool(backend)`, not the device-less `ActivePool()` this line was
        // written against (#516): "the pool" without a device is what handed one
        // device's block to another, and draining it through `backend` was the
        // same assumption twice — resolve a pool with no device, then free its
        // blocks through an allocator that may not have made them. There is no
        // device-less spelling any more, and `Drain` refuses a foreign backend.
        const size_t drained = ActivePool(backend).Drain(backend);
        if (std::getenv("VT_POOL_STATS") != nullptr) {
          std::fprintf(stderr, "[ltx2] phase '%s' drained %.2f GiB of denoise scratch\n",
                       phase.name.c_str(),
                       static_cast<double>(drained) / (1024.0 * 1024.0 * 1024.0));
        }
      }
    }
  }

  phase::Scope trim_phase("generate.trim");
  // ── DFR: trim the padded canvas back to the caller's contract (#986) ──────
  //
  // `dfr_pipeline.py:531-540`. The canvas padded its tail up to a whole number
  // of keyframe segments, and what the caller is owed is
  // `(requested - 1) * 2**rounds + 1`. With the rounds arm refused above, that
  // is `requested` itself — written through `Ltx2DfrTargetFrames` anyway rather
  // than as `requested_frames`, so the two expressions cannot drift apart when
  // the rounds land.
  //
  // `requested - 1` is a multiple of the VAE temporal scale — `resolve_canvas`
  // refuses anything else (dfr_layout.py:71-72) — so the trim always lands on a
  // latent boundary and is a SLICE rather than a resample. Upstream RAISES when
  // the target exceeds the canvas (:535-536), which is a statement about the
  // arithmetic rather than a defensive check: the canvas only ever pads UP.
  if (is_dfr) {
    const int64_t target_frames = Ltx2DfrTargetFrames(requested_frames, /*rounds=*/0);
    if (target_frames > frames) {
      Fail("target " + std::to_string(target_frames) + " frames exceeds the generated canvas " +
           std::to_string(frames) +
           " (ltx-pipelines/dfr_pipeline.py:535-536). The canvas only ever pads UP, so this "
           "means the segment grid and the trim disagree.");
    }
    if (target_frames != frames) {
      const int64_t keep = (target_frames - 1) / factors.time + 1;
      if (keep > video_lf) {
        Fail("the DFR trim wants " + std::to_string(keep) + " latent frames and the canvas has " +
             std::to_string(video_lf));
      }
      // [C, F, H, W] row-major, so the trim is a per-CHANNEL copy of the leading
      // `keep` frames rather than a truncation of the flat buffer. A plain
      // resize would keep channel 0's whole time axis and drop the last
      // channel's entirely, and the result is still a correctly shaped latent.
      const int64_t plane = video_lh * video_lw;
      std::vector<float> trimmed(static_cast<size_t>(video_lc * keep * plane));
      for (int64_t c = 0; c < video_lc; ++c) {
        const size_t src = static_cast<size_t>(c * video_lf * plane);
        std::copy(video_latent_volume.begin() + static_cast<ptrdiff_t>(src),
                  video_latent_volume.begin() + static_cast<ptrdiff_t>(src) +
                      static_cast<ptrdiff_t>(keep * plane),
                  trimmed.begin() + static_cast<ptrdiff_t>(c * keep * plane));
      }
      video_latent_volume.swap(trimmed);
      video_lf = keep;
      frames = target_frames;
    }
    // The audio was generated for the PADDED canvas, so it outlasts the picture
    // and upstream cuts it to the video's duration (dfr_pipeline.py:552-560).
    // That cut is done AFTER the vocoder, beside the waveform it applies to;
    // see the block above `VideoResult result` at the end of this function.
    //
    // This comment used to say the cut was "already implied by the trimmed
    // `frames`" and that was FALSE: `audio_lf` carries the padded count out of
    // the phase loop and the vocoder runs over all of it, so trimming the video
    // latent here changes nothing about the soundtrack.
  }

  // ── decode (distilled.py:314-315) ─────────────────────────────────────────
  //
  // STREAMED, through upstream's own AUTO tiling, mirroring
  // `ti2vid_two_stages.py:365-376`: the pipeline passes `AUTO_TILING` and the
  // consumer takes chunks straight out of `decode_video` instead of a whole clip.
  //
  // AT THE SIZES THIS PROJECT HAS RUN, THIS CHANGES NOTHING, and that is a
  // measured statement rather than a hope. `Ltx2AutoTileSizeConfig` is upstream's
  // Conv layout (768px tile / 64px overlap on the long side, 80 frames / 24
  // overlap — helpers.py:59-88), and `split_by_size` returns ONE interval when the
  // axis is no bigger than the tile (tiling.py:199-200). At 448x256/25f the latent
  // is 8x14 against a 14x24 grid tile and 4 frames against a 10-latent-frame
  // temporal tile, so exactly one tile and one chunk come out — and the ONE-TILE
  // CONTROL in tests/vllm/models/test_ltx2_tiling.cpp proves that path reproduces
  // the untiled decode BIT FOR BIT (max|diff| == 0, on both causality arms, and
  // upstream's own value for the same control is 0 too).
  //
  // TILING FIRST DOES ANYTHING AT 896x512 SPATIALLY AND AT **81 FRAMES**
  // TEMPORALLY — 81, not 121. `latent_t = (frames - 1) / 8 + 1` is 11 at 81
  // frames and `split_temporal_causal` short-circuits only while `latent_t <= 10`
  // (tiling.py:239-240). The row's own golden `kLtx2AutoCases` has said so since
  // it was generated (`768x768/81f -> t_intervals = 2`); the prose here said 121
  // because the probe sweep that produced it stepped 25 -> 121 and never sampled
  // the binding point.
  //
  // SO THE "CHANGES NOTHING" ABOVE IS BOUNDED BY 81 FRAMES, AND A DEFAULT REQUEST
  // IS NOT INSIDE THAT BOUND. `docs/USAGE.md` records the recipe default as
  // 1024x1536 at 121 frames. Between 81 and 120 frames the render is tiled and is
  // NOT the render this path produced before tiling existed — measured on the
  // SHIPPED conv VAE at the AUTO layout, 64x64 / 81 frames, latent 11,2,2, by
  // scripts/probe_ltx2_tiled_equivalence.cpp: 2 tiles, 2 chunks, max|diff|
  // 0.0503043234 against the untiled decode on an output whose own |max| is
  // 0.7512672544 — 6.70% of that range — with 962983 of 995328 floats (96.75%)
  // not bit-identical. That is upstream's own behaviour (a receptive field wider
  // than the overlap, blended at the seam) and not a defect here — but it is a
  // different image, and the one-tile control does not cover it.
  //
  // (This said 0.716 and "95% of the range" until the probe was re-derived: it
  // reassembled the streamed chunks with a FLAT append, and a chunk is
  // [C, t, H, W] channel-major, so the append is not [C, T, H, W] at C = 3 with
  // 2 chunks. The 14x was the probe's own transposition. The conclusion stands,
  // the magnitude did not — see ltx2_tiling.h for the full record.)
  //
  // What it buys, ABOVE ONE CHUNK: the full pixel volume is never materialized.
  // Each temporal chunk is written to disk and dropped, so the peak is about two
  // chunks rather than [3, F, H, W] — which is what makes the long clips
  // upstream's 80/24 layout exists for reachable at all. At exactly one chunk the
  // chunk IS the volume and nothing is saved; that is the regime below 81 frames,
  // and it is no worse than the buffered path it replaced.
  EngineNoiseStream decode_noise(seed ^ 0x1D7Cull);
  const Ltx2ScaleFactors video_factors =
      Ltx2VideoScaleFactorsFromBlocks(im.video_cfg.decoder_blocks, im.video_cfg.patch_size);
  const int64_t rendered_h = video_lh * video_factors.height;
  const int64_t rendered_w = video_lw * video_factors.width;

  // ── artifacts (the library WRITES these, spawns nothing) ──────────────────
  // Hoisted ABOVE the decode because the decode now writes into it chunk by
  // chunk; a directory created after the first chunk arrived would be too late.
  std::error_code ec;
  std::filesystem::create_directories(gen.output_dir, ec);
  if (ec) Fail("cannot create " + gen.output_dir + ": " + ec.message());

  // A PREVIOUS RENDER'S TAIL IS DELETED BEFORE THIS ONE STARTS, and it has to be.
  //
  // The muxer is handed `frame_%06d.ppm` with no frame count (see `mux.frame_pattern`
  // below), so it takes whatever consecutive files it finds. A 121-frame render
  // followed by a 25-frame render into the same directory would leave
  // frame_000025..frame_000120 on disk and mux a clip that runs 96 frames past its
  // own end — silently, and looking like the longer render succeeded. Streaming
  // widened this: a chunk that throws now also leaves a partial render behind,
  // where the old buffered path wrote nothing until the whole decode had finished.
  //
  // Scoped deliberately: only `frame_<digits>.ppm`, only in the directory this
  // render is about to write, and nothing else in it is touched. Collected first
  // and removed after, because unlinking the entry the iterator is standing on is
  // not something the directory iterator promises to survive.
  {
    std::vector<std::filesystem::path> stale;
    for (const std::filesystem::directory_entry& entry :
         std::filesystem::directory_iterator(gen.output_dir, ec)) {
      if (ec) break;
      if (!entry.is_regular_file()) continue;
      const std::string name = entry.path().filename().string();
      // "frame_" + at least one digit + ".ppm" is 11 characters.
      if (name.size() < 11) continue;
      if (name.rfind("frame_", 0) != 0) continue;
      if (name.compare(name.size() - 4, 4, ".ppm") != 0) continue;
      bool all_digits = true;
      for (size_t i = 6; i + 4 < name.size(); ++i) {
        if (std::isdigit(static_cast<unsigned char>(name[i])) == 0) all_digits = false;
      }
      if (!all_digits) continue;
      stale.push_back(entry.path());
    }
    for (const std::filesystem::path& p : stale) {
      std::error_code rm;
      std::filesystem::remove(p, rm);
    }
    ec.clear();
  }

  trim_phase.Close();

  int64_t rendered_frames = 0;
  int64_t rendered_channels = 0;
  // W0: the decode and the PPM writer INTERLEAVE — the decoder streams chunks
  // into the writer's callback — so the two leaves ALTERNATE rather than nest.
  // Folding the writes into `decode.video` would charge W5's lever with the cost
  // of a `write(2)`; nesting them would take the writer out of the sum entirely.
  size_t decode_handle = phase::PhaseLog::Instance().Open("decode.video", /*span=*/false);
  // W0 repair (#1010, second fresh review): the same anchor on the video side.
  // `decode.video.chunk` opens with the leaf and closes when the decoder hands a
  // chunk BACK, so its end is a production event rather than an instrument
  // statement. A `decode.video` leaf that closes before its chunk arrives, or
  // that is re-labelled after one, no longer contains the chunk it produced.
  // Nested, so the sum does not move.
  size_t chunk_handle =
      phase::PhaseLog::Instance().Open("decode.video.chunk", /*span=*/false);
  Ltx2VideoDecodeStreaming(
      im.video_kind, im.video_cfg, im.video_weights, video_latent_volume, video_lc, video_lf,
      video_lh, video_lw, &decode_noise,
      Ltx2AutoTileSizeConfig(rendered_h, rendered_w, video_factors),
      [&](const Ltx2VideoChunk& chunk) {
        phase::PhaseLog::Instance().Close(chunk_handle);
        phase::PhaseLog::Instance().Close(decode_handle);
        phase::Scope write_phase("artifacts.frames");
        MiniMaxH3VideoFrameShape shape;
        shape.channels = chunk.frames.channels;
        shape.t = chunk.frames.frames;
        shape.h = chunk.frames.height;
        shape.w = chunk.frames.width;
        for (int64_t f = 0; f < chunk.frames.frames; ++f) {
          char name[64];
          // The GLOBAL frame index, which the chunk carries so the writer does not
          // have to count. Numbering per-chunk would silently reorder the clip.
          std::snprintf(name, sizeof(name), "frame_%06lld.ppm",
                        static_cast<long long>(chunk.first_frame + f));
          // The PPM/WAV/mux writers are the SHARED serialization the spec's §5
          // reuse list names, not H3 behaviour: they take a buffer and a shape and
          // contain no model. Reimplementing them here would be the parallel path
          // §"Shared seams" forbids.
          WriteFileBytes(JoinPath(gen.output_dir, name),
                         MiniMaxH3WritePpmFrame(chunk.frames.data, shape, f));
        }
        rendered_frames += chunk.frames.frames;
        rendered_channels = chunk.frames.channels;
        write_phase.Close();
        decode_handle = phase::PhaseLog::Instance().Open("decode.video", /*span=*/false);
        chunk_handle = phase::PhaseLog::Instance().Open("decode.video.chunk", /*span=*/false);
      });
  phase::PhaseLog::Instance().Close(chunk_handle);
  phase::PhaseLog::Instance().Close(decode_handle);

  // ── the soundtrack ────────────────────────────────────────────────────────
  //
  // On the AUDIO-TO-VIDEO path the caller's own take is returned UNCHANGED, and
  // the audio VAE decode and the vocoder do not run at all. Upstream says why in
  // as many words: "Return the original input audio instead of VAE-decoded audio
  // to preserve fidelity" (a2vid_two_stage.py:301-303). The latent was frozen
  // all the way through, so the reconstruction could at best equal the file the
  // caller already has, and in practice loses an encode and a vocoder pass to
  // it. Skipping the chain is upstream's behaviour, not an optimisation.
  int64_t audio_channels = 0;
  int64_t audio_samples = 0;
  int64_t audio_rate = 0;
  std::vector<float> waveform;
  if (!a2v_audio_volume.empty()) {
    waveform = a2v_source.samples;
    audio_channels = a2v_source.channels;
    audio_samples = a2v_source.samples_per_channel;
    audio_rate = a2v_source.sample_rate;
  } else {
    // W0 repair (#1010 review): `decode.audio` FOLDED TWO MODELS, and one of
    // them was the second-largest phase in the first artifact this row shipped
    // — 3.062 s, 25.5% of wall, un-decomposed. #1010 asks for the vocoder by
    // name. The two are split as NESTED leaves inside the same `decode.audio`
    // leaf rather than as siblings: nested records are excluded from the sum
    // (`render_phase_log.cpp` `Sum`), so the decomposition is readable without
    // changing what the table adds up to, and `decode.audio` stays the boundary
    // the gate requires.
    const phase::Scope audio_decode_phase("decode.audio");
    phase::Scope mel_phase("decode.audio.mel");
    const Ltx2AudioSpectrogram mel = Ltx2AudioDecoderForward(
        im.audio_cfg, im.audio_weights, audio_latent_volume, audio_lc, audio_lf, audio_lm);
    mel_phase.Close();
    phase::Scope vocoder_phase("decode.audio.vocoder");
    waveform = Ltx2VocoderWithBweForward(im.vocoder_cfg, im.vocoder_weights, mel.data,
                                         mel.channels, mel.frames, mel.mel_bins, &audio_samples);
    vocoder_phase.Close();
    audio_channels = mel.channels;
    audio_rate = im.vocoder_cfg.output_sampling_rate;
  }

  // ── DFR: cut the soundtrack to the picture (#986) ─────────────────────────
  //
  // `dfr_pipeline.py:552-560`, and upstream states the consequence rather than
  // the mechanism: "Audio was generated for the padded canvas, so cut it to the
  // video's duration or the muxed container outlasts the picture."
  //
  // THE AUDIO IS NOT COVERED BY THE VIDEO TRIM ABOVE, and a comment here claimed
  // it was until this was checked. `ashape.frames` is derived from the PADDED
  // `frames` inside the phase loop, `audio_lf` carries that padded count out of
  // the loop, and the vocoder above runs over all of it. Trimming the video
  // latent moves `frames` and touches none of that, so a 9-frame DFR request
  // would emit 9 frames of picture beside 25 frames' worth of sound. Nothing in
  // the render's shape, its frame count or its exit status can see it; it shows
  // up only in the muxed container, which this library does not produce.
  //
  // `min` because the waveform may already be shorter — upstream takes the same
  // min, and a cut that grew the buffer would read past its end.
  if (is_dfr && audio_rate > 0 && audio_samples > 0) {
    const double video_seconds = static_cast<double>(frames) / fps;
    const int64_t want = std::min<int64_t>(
        audio_samples, static_cast<int64_t>(std::llround(video_seconds *
                                                         static_cast<double>(audio_rate))));
    if (want > 0 && want != audio_samples) {
      // The waveform is [channels, samples_per_channel] and the cut is per
      // CHANNEL, which is upstream's `waveform[..., :audio_samples]`. A flat
      // resize would keep channel 0 whole and truncate the last one to nothing,
      // and the result is still a playable file.
      std::vector<float> cut(static_cast<size_t>(audio_channels * want));
      for (int64_t c = 0; c < audio_channels; ++c) {
        const size_t src = static_cast<size_t>(c * audio_samples);
        std::copy(waveform.begin() + static_cast<ptrdiff_t>(src),
                  waveform.begin() + static_cast<ptrdiff_t>(src) + static_cast<ptrdiff_t>(want),
                  cut.begin() + static_cast<ptrdiff_t>(c * want));
      }
      waveform.swap(cut);
      audio_samples = want;
    }
  }

  VideoResult result;
  result.frame_dir = gen.output_dir;
  // The streamed chunks must have covered exactly the clip the latent implies. A
  // decode that dropped or duplicated a temporal group would otherwise show up
  // only as a short mp4.
  const int64_t expect_frames = (video_lf - 1) * video_factors.time + 1;
  if (rendered_frames != expect_frames || rendered_channels != im.video_cfg.out_channels) {
    Fail("the streamed video decode produced " + std::to_string(rendered_frames) + " frames x " +
         std::to_string(rendered_channels) + " channels, but the latent implies " +
         std::to_string(expect_frames) + " x " + std::to_string(im.video_cfg.out_channels));
  }
  result.audio_path = JoinPath(gen.output_dir, "audio.wav");
  {
    const phase::Scope wav_phase("artifacts.audio");
    WriteFileBytes(result.audio_path,
                   MiniMaxH3WriteWav(waveform, audio_channels, audio_samples, audio_rate));
  }
  result.frame_count = rendered_frames;
  result.width = rendered_w;
  result.height = rendered_h;
  result.fps = static_cast<int64_t>(std::llround(fps));
  result.sample_rate = audio_rate;

  MiniMaxH3MuxRequest mux;
  mux.frame_pattern = JoinPath(gen.output_dir, "frame_%06d.ppm");
  mux.audio_path = result.audio_path;
  mux.output_path = JoinPath(gen.output_dir, "video.mp4");
  mux.fps = result.fps;
  result.mux_argv = MiniMaxH3BuildMp4MuxArgs(mux);
  result.mux_output_path = mux.output_path;
  // The trace was filled before the denoise loop, which is the only place the
  // buffers cross-attention reads still exist as such. Everything between there
  // and here can throw, so `completed` is set HERE and nowhere else: it is what
  // separates "this conditioning produced that clip" from "this conditioning was
  // built for a render that then failed".
  im.trace.completed = true;
  // W0 (#1010): the table, beside the frames it explains. The enclosing span is
  // closed BEFORE the write so this render's own span appears in its own file.
  generate_span.Close();
  WritePhaseLog(gen.output_dir, kLtx2VideoFamily, phase_device, &result.phase_log_path);
  return result;
}

// ── TEXT-TO-AUDIO (row LTX25-T2A-ONE-STAGE, #1005) ──────────────────────────
//
// `T2AOneStagePipeline.__call__` (ltx-pipelines t2a_one_stage.py:109-172). The
// numerics live in `ltx2_t2a.cpp`, mirroring upstream's own file; this resolves
// the request, encodes the negative prompt and writes the artifact.
//
// PLACED BELOW `Generate` ON PURPOSE. The READER ANCHORS comment near the top of
// this file carries derived LINE NUMBERS into it and is gated by
// `test_ltx2_video`, so a definition inserted above the last anchored line would
// move every anchor under it for a reason that has nothing to do with this row.
VideoResult Ltx2VideoEngine::GenerateAudioOnly(Impl& im, const VideoGenParams& gen,
                                               const float* audio_context,
                                               int64_t context_tokens) {
  const Ltx2PipelineRecipe& recipe = im.recipe;
  VT_CHECK(recipe.audio_only, "ltx2 t2a: reached the audio-only path on a video recipe");
  // This path does NOT run the phase loop — it reads `phases.front()` and
  // denoises once — so it is the one place a per-phase adapter set could be
  // declared and silently ignored. Upstream's `T2AOneStagePipeline` builds ONE
  // stage set (t2a_one_stage.py:67), so no audio-only recipe has a reason to ask
  // for anything but the load's own adapters, and this asserts that rather than
  // assuming it. Row LTX25-PHASE-LORA (#1118).
  VT_CHECK(recipe.phases.front().loras == Ltx2PhaseLoraScope::kAllAdapters,
           "ltx2 t2a: an audio-only phase asked for a per-phase adapter set, and this path "
           "never runs the rebind that would honour it");

  // Upstream's T2A CLI has no --height/--width, and its pipeline substitutes a
  // 512x512 PLACEHOLDER whose height and width it documents as unused
  // (t2a_one_stage.py:37-40, :163-164). Accepting a resolution here would take a
  // number from the caller, ignore it, and return successfully.
  if (gen.height > 0 || gen.width > 0) {
    Fail("a text-to-audio request cannot carry a width or a height: there is no picture. "
         "Upstream passes a 512x512 PLACEHOLDER into the stage and says so in as many words — "
         "\"Audio-only generation reads `frames` and `fps` from the pixel shape via "
         "`AudioLatentShape.from_video_pixel_shape` (height/width are unused)\" "
         "(t2a_one_stage.py:37-40). Accepting one would ignore it and still succeed");
  }
  if (!gen.first_frame_path.empty() || !gen.last_frame_path.empty() ||
      !gen.first_frame_ppm.empty() || !gen.ref_image_paths.empty() ||
      !gen.ref_video_dir.empty() || !gen.ref_audio_path.empty() || !gen.ref_audio_wav.empty()) {
    Fail("a text-to-audio request cannot carry a keyframe, a reference image, a reference clip "
         "or a reference waveform. `T2AOneStagePipeline.__call__` takes none of them "
         "(t2a_one_stage.py:109-122) and its `DiffusionStage` call passes `video=None` "
         "(`:167`), so there is no stream for any of them to condition");
  }

  // `num_frames` / `frame_rate` — the only two fields of the placeholder pixel
  // shape T2A reads, and they exist to derive the audio DURATION.
  const double fps = recipe.frame_rate;
  int64_t frames = gen.num_frames > 1 ? gen.num_frames : recipe.num_frames;
  if (gen.duration_seconds > 0.0) {
    frames = static_cast<int64_t>(std::llround(gen.duration_seconds * fps));
  }

  // ── the guider (t2a_one_stage.py:196-205) ─────────────────────────────────
  //
  // The recipe already carries the params table's audio guider with
  // `modality_scale` pinned to 1.0; each extra overrides ONE field, exactly as
  // one CLI flag does.
  Ltx2MultiModalGuiderParams guidance = recipe.phases.front().audio_guidance;
  guidance.cfg_scale = ExtraDouble(gen.extras, kLtx2AudioCfgScaleExtra, guidance.cfg_scale);
  guidance.stg_scale = ExtraDouble(gen.extras, kLtx2AudioStgScaleExtra, guidance.stg_scale);
  guidance.rescale_scale =
      ExtraDouble(gen.extras, kLtx2AudioRescaleScaleExtra, guidance.rescale_scale);
  guidance.skip_step = ExtraInt(gen.extras, kLtx2AudioSkipStepExtra, guidance.skip_step);
  if (guidance.skip_step < 0) {
    Fail("'" + std::string(kLtx2AudioSkipStepExtra) + "' is " +
         std::to_string(guidance.skip_step) +
         "; `should_skip_step` is `step % (skip_step + 1)` (guiders.py:287-291) and a negative "
         "value would take the modulus of a non-positive divisor");
  }
  {
    // `--audio-stg-blocks`, `nargs="*"` (utils/args.py:1107-1113). An extra that
    // is PRESENT and empty is upstream's empty list — "perturb nothing" — and is
    // kept distinct from an ABSENT extra, which takes the params table's own
    // [28]. Collapsing the two would make `audio_stg_blocks=` silently mean
    // block 28.
    const auto at = gen.extras.find(kLtx2AudioStgBlocksExtra);
    if (at != gen.extras.end()) {
      guidance.stg_blocks.clear();
      const std::string& raw = at->second;
      for (size_t i = 0; i < raw.size();) {
        const size_t comma = raw.find(',', i);
        const std::string token = raw.substr(i, comma == std::string::npos ? comma : comma - i);
        if (!token.empty()) {
          try {
            guidance.stg_blocks.push_back(std::stoll(token));
          } catch (const std::exception&) {
            Fail("'" + std::string(kLtx2AudioStgBlocksExtra) + "' holds '" + token +
                 "', which is not an integer block index");
          }
        }
        if (comma == std::string::npos) break;
        i = comma + 1;
      }
    }
  }

  // ── the negative conditioning (t2a_one_stage.py:127-135) ──────────────────
  //
  // Upstream encodes `[prompt, negative_prompt]` in ONE `PromptEncoder` call and
  // takes `.audio_encoding` from each. Here the positive half was already
  // resolved by `Generate`; this is the second half, through the same
  // `Ltx2EncodePromptToConditioning` and the same connector.
  //
  // ONLY WHEN THE GUIDER ASKS FOR IT. `do_unconditional_generation` is
  // `not isclose(cfg_scale, 1.0)` (guiders.py:275-277), so at scale 1.0 there is
  // no unconditional forward and encoding a negative prompt would be a wasted
  // 12B host-side pass per request.
  std::vector<float> negative_audio;
  const float* negative_context = nullptr;
  if (!guidance.DoUnconditionalGeneration()) {
    // Nothing to do: the guidance delta's `uncond_text` term is switched off.
  } else if (!im.has_encoder) {
    Fail("this text-to-audio request needs an unconditional forward (`cfg_scale` = " +
         std::to_string(guidance.cfg_scale) +
         "), which needs the NEGATIVE prompt encoded — and no text tower is loaded, so this "
         "engine can encode neither prompt. The `" +
         std::string(kLtx2AudioPromptEmbedsExtra) +
         "' fallback carries ONE conditioning stream and there is no second file for the "
         "negative one. Load with encoder_path, or set '" +
         std::string(kLtx2AudioCfgScaleExtra) +
         "' to 1.0, which turns the unconditional pass off (guiders.py:275-277)");
  } else {
    const std::string negative =
        VideoExtra(gen.extras, kLtx2NegativePromptExtra, recipe.negative_prompt);
    if (negative.empty()) {
      Fail("this text-to-audio request needs a negative prompt and neither the '" +
           std::string(kLtx2NegativePromptExtra) +
           "' extra nor the recipe carries one. An EMPTY negative prompt is not the same as no "
           "CFG: it still encodes and still steers, and upstream's CLI always supplies "
           "`DEFAULT_NEGATIVE_PROMPT` (utils/args.py:1083-1088)");
    }
    vt::Queue text_queue{vt::Device{vt::DeviceType::kCPU, 0}, nullptr};
    const Ltx2PromptConditioning encoded = Ltx2EncodePromptToConditioning(
        *im.tower, *im.tokenizer, im.gemma_ids, im.caption_projections, im.feature_cfg, negative,
        text_queue);
    negative_audio = encoded.conditioning.audio;
    if (encoded.seq != context_tokens) {
      // Upstream's two encodings come from ONE tokenization of a two-element
      // list (t2a_one_stage.py:127-133), so they share a padded width by
      // construction. A mismatch here means the two ran different geometries and
      // the guidance delta would subtract tensors that do not correspond.
      Fail("the negative prompt encoded to " + std::to_string(encoded.seq) +
           " context rows and the prompt to " + std::to_string(context_tokens) +
           "; upstream encodes both in one call and they cannot differ");
    }
    if (im.has_connector) {
      const Ltx2ConnectorEmbeddings through =
          RunConnector(SafetensorsFile::Open(im.params.dit_path), im.video_connector_cfg,
                       im.audio_connector_cfg, encoded.conditioning.video,
                       encoded.conditioning.audio, encoded.conditioning.additive_mask,
                       context_tokens);
      negative_audio = through.audio;
    }
    negative_context = negative_audio.data();
  }

  // ── the render ────────────────────────────────────────────────────────────
  if (im.on_device) {
    // REFUSED BY NAME rather than served the host forward behind a device
    // handle, which is the substitution this engine's header names as the thing
    // that would make every later timing claim false.
    //
    // WHAT IS *NOT* THE REASON: not the STG perturbation and not the guider.
    // `Ltx2DitPerturbation` is a plain argument either forward could take, and
    // `Ltx2MultiModalGuidance` runs on host buffers on both arms. What is
    // missing is narrower and is a fact about THIS tree:
    // `Ltx2DitForwardDevice` dereferences `*video` unconditionally — both
    // `PrepareStreamDev` calls take it by reference, and the per-block
    // `a.batch = video->batch` reads through it — so a one-stream device forward
    // is a rewrite of that function rather than the lifted check the host
    // forward needed. Owed by #1005.
    Fail("text-to-audio is not served on the accelerator. `Ltx2DitForwardDevice` takes BOTH "
         "streams by reference and this pipeline has no video stream to give it "
         "(`video=None`, t2a_one_stage.py:167). Refusing rather than running the host forward "
         "behind a device handle. Load with device 0.");
  }

  EngineNoiseStream noise(gen.has_seed ? gen.seed : static_cast<uint64_t>(recipe.num_frames));
  Ltx2T2aRequest req;
  req.device = im.device;
  req.compute_dtype = im.compute_dtype;
  req.dit_params = &im.dit.params;
  req.dit_weights = &im.dit.weights;
  req.context = audio_context;
  req.negative_context = negative_context;
  req.context_tokens = context_tokens;
  req.num_frames = frames;
  req.frame_rate = fps;
  req.steps = gen.steps > 0 ? gen.steps : recipe.num_inference_steps;
  req.noise = &noise;
  req.guidance = guidance;
  req.audio_cfg = &im.audio_cfg;
  req.audio_weights = &im.audio_weights;
  req.vocoder_cfg = &im.vocoder_cfg;
  req.vocoder_weights = &im.vocoder_weights;

  const Ltx2T2aResult rendered = Ltx2T2aGenerate(req);

  im.trace.t2a_rendered = true;
  im.trace.t2a_video_stream_present = rendered.video_stream_present;
  im.trace.t2a_cond_forwards = rendered.cond_forwards;
  im.trace.t2a_uncond_forwards = rendered.uncond_forwards;
  im.trace.t2a_perturbed_forwards = rendered.perturbed_forwards;
  im.trace.t2a_perturbed_blocks = rendered.perturbed_blocks;
  im.trace.t2a_first_latent = rendered.first_step_latent;
  im.trace.t2a_first_velocity = rendered.first_step_velocity;
  im.trace.t2a_first_cond = rendered.first_step_cond;
  im.trace.t2a_first_uncond_velocity = rendered.first_step_uncond_velocity;
  im.trace.t2a_first_uncond = rendered.first_step_uncond;
  im.trace.t2a_first_perturbed_velocity = rendered.first_step_perturbed_velocity;
  im.trace.t2a_first_perturbed = rendered.first_step_perturbed;
  im.trace.t2a_first_denoised = rendered.first_step_denoised;
  im.trace.t2a_first_next_latent = rendered.first_step_next_latent;
  im.trace.t2a_first_sigma = rendered.first_step_sigma;
  im.trace.audio_tokens = rendered.audio_tokens;
  im.trace.audio_latent_digest = rendered.latent_digest;
  im.trace.audio_latent_absmax = rendered.latent_absmax;

  // ── the artifact ──────────────────────────────────────────────────────────
  std::error_code ec;
  std::filesystem::create_directories(gen.output_dir, ec);
  if (ec) Fail("cannot create " + gen.output_dir + ": " + ec.message());

  VideoResult result;
  // `frame_dir` STAYS EMPTY, and that is the contract rather than an omission: a
  // directory naming a frame pattern that matches no file is what a caller
  // iterates and finds nothing in. `frame_count == 0` says the same thing, and
  // saying it twice is what lets a consumer notice the disagreement if one of
  // them is ever filled by accident.
  result.frame_count = 0;
  result.width = 0;
  result.height = 0;
  result.fps = 0;
  result.audio_path = JoinPath(gen.output_dir, "audio.wav");
  WriteFileBytes(result.audio_path,
                 MiniMaxH3WriteWav(rendered.waveform, rendered.channels,
                                   rendered.samples_per_channel, rendered.sample_rate));
  result.sample_rate = rendered.sample_rate;
  // NO `mux_argv`, and none is composed. The argv this seam builds muxes a frame
  // pattern with a soundtrack; over an empty pattern ffmpeg fails, so composing
  // one would hand the caller a command that cannot run and call it a result.
  im.trace.completed = true;
  return result;
}

namespace {

std::unique_ptr<VideoEngine> LoadLtx2VideoFamily(const VideoModelParams& params) {
  return Ltx2VideoEngine::Load(params);
}

}  // namespace

REGISTER_VLLM_VIDEO_FAMILY(ltx2, kLtx2VideoFamily, DetectLtx2Video, LoadLtx2VideoFamily)

}  // namespace vllm::multimodal
