// LTX-2.5 behind the generalized video seam — implementation. See
// include/vllm/multimodal/ltx2_video.h for the port map and the three refusals.
//
// Row: MODEL-DIFFUSION-LTX25, .agents/specs/ltx-2-5.md phase L7. Issue #435.
#include "vllm/multimodal/ltx2_video.h"

#include <algorithm>
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
#include "vllm/model_executor/models/ltx2_device.h"
#include "vllm/model_executor/models/ltx2_image_preprocess.h"
#include "vllm/model_executor/models/ltx2_loader.h"
#include "vllm/model_executor/models/ltx2_pipeline.h"
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
  out.positions.assign(s.positions.begin(), s.positions.end());
  out.keyframes_mask = s.keyframes_mask;
  return out;
}

void FromLatentState(const Ltx2LatentState& in, StreamState* s) {
  s->tokens = in.tokens;
  s->width = in.width;
  s->latent = in.latent;
  s->clean = in.clean;
  s->mask = in.mask;
  s->positions.assign(in.positions.begin(), in.positions.end());
  s->keyframes_mask = in.keyframes_mask;
}

// `post_process_latent` (utils/helpers.py:462-464):
//   denoised * mask + clean * (1 - mask)
// The mask is PER TOKEN and the latent is per token x channel, so the mask
// broadcasts along the channel axis exactly as torch's trailing-axis rule does.
std::vector<float> PostProcessLatent(const std::vector<float>& denoised, const StreamState& state) {
  std::vector<float> out(denoised.size());
  for (int64_t t = 0; t < state.tokens; ++t) {
    const float m = state.mask[static_cast<size_t>(t)];
    for (int64_t c = 0; c < state.width; ++c) {
      const size_t i = static_cast<size_t>(t * state.width + c);
      out[i] = denoised[i] * m + state.clean[i] * (1.0F - m);
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

std::string JoinPath(const std::string& dir, const std::string& leaf) {
  if (dir.empty()) return leaf;
  return dir.back() == '/' ? dir + leaf : dir + "/" + leaf;
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
// 756 811 907 923 925 1003 1028 1133 1174
const char* const kKnownLoadExtras[] = {
    kLtx2AudioPromptEmbedsExtra, kLtx2PipelineKindExtra,   kLtx2ModelVersionExtra,
    kLtx2AllowUnportedExtra,     kLtx2MaxPhaseExtra,       kLtx2DitConfigPathExtra,
    kLtx2PromptValidRowsExtra,   kLtx2EncoderConfigPathExtra,
    "upsampler_path",            kLtx2DurationHeadPathExtra,
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
  // `SelectQueueForModel` (src/vllm/entrypoints/model_loader.cpp:75-104 — the
  // full path matters, there is also a src/vllm/model_executor/model_loader/
  // DIRECTORY and the bare file name sends a reader there), where a hardcoded
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
  Ltx2DitLoadOptions dit_options;
  dit_options.allow_unported_modules = VideoExtra(params.extras, kLtx2AllowUnportedExtra) == "1";
  // On the CPU, f32 is what `Ltx2DitForward` requires: it is the PARITY dtype,
  // not a widening of a bf16 path. On an accelerator nothing is widened at all —
  // `Ltx2StreamDitToDevice` dequantizes and uploads ONE TENSOR AT A TIME, so peak
  // residency is the device copy plus one tensor rather than two whole models.
  dit_options.widen_to_f32 = !im.on_device;
  im.dit = im.on_device ? Ltx2StreamDitToDevice(*im.queue, dit_file, dit_options)
                        : Ltx2LoadDitFromSafetensors(dit_file, dit_options);

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
  im.max_phase = ExtraInt(params.extras, kLtx2MaxPhaseExtra, -1);
  if (im.max_phase >= static_cast<int64_t>(im.recipe.phases.size())) {
    Fail("the '" + std::string(kLtx2MaxPhaseExtra) + "' extra is " +
         std::to_string(im.max_phase) + " but the '" + im.pipeline_kind + "'/'" +
         im.model_version + "' recipe has " + std::to_string(im.recipe.phases.size()) +
         " phases");
  }

  // ── the video VAE ─────────────────────────────────────────────────────────
  if (params.video_vae_path.empty()) Fail("video_vae_path is required");
  {
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
  if (im.video_cfg.in_channels != im.dit.params.out_channels) {
    Fail("the video VAE takes " + std::to_string(im.video_cfg.in_channels) +
         " latent channels but the DiT emits " + std::to_string(im.dit.params.out_channels));
  }

  // ── the audio VAE + its vocoder ───────────────────────────────────────────
  if (params.audio_vae_path.empty()) Fail("audio_vae_path is required");
  {
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
  }
  return engine;
}

VideoResult Ltx2VideoEngine::Generate(const VideoGenParams& gen) {
  Impl& im = *impl_;
  std::lock_guard<std::mutex> guard(im.mutex);

  if (gen.output_dir.empty()) Fail("output_dir is required");
  for (const auto& kv : gen.extras) {
    // `image_crf` is the only per-generation extra this family defines (row
    // LTX25-IMAGE-COND). Everything else is refused rather than ignored, for the
    // reason `CheckKnownExtras` gives for the load side: a mistyped knob that is
    // silently dropped renders the DEFAULT and looks like the feature not
    // working — and for THIS knob the default is a refusal, so a typo would turn
    // a served request into an unexplained one.
    const bool known = kv.first == kLtx2ImageCrfExtra || kv.first == kLtx2AudioPathExtra ||
                       kv.first == kLtx2AudioStartTimeExtra ||
                       kv.first == kLtx2AudioMaxDurationExtra;
    if (!known) {
      Fail("unknown per-generation extra '" + kv.first + "'. This family defines: " +
           std::string(kLtx2ImageCrfExtra) + ", " + kLtx2AudioPathExtra + ", " +
           kLtx2AudioStartTimeExtra + ", " + kLtx2AudioMaxDurationExtra);
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
  std::vector<float> prompt_video, prompt_audio;
  const float* video_context = im.video_prompt_embeds.data();
  const float* audio_context = im.audio_prompt_embeds.data();
  int64_t context_tokens = im.prompt_tokens;
  im.trace = Ltx2ConditioningTrace{};

  if (!gen.prompt.empty()) {
    vt::Queue text_queue{vt::Device{vt::DeviceType::kCPU, 0}, nullptr};
    const Ltx2PromptConditioning encoded = Ltx2EncodePromptToConditioning(
        *im.tower, *im.tokenizer, im.gemma_ids, im.caption_projections, im.feature_cfg,
        gen.prompt, text_queue);
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
      const Ltx2ConnectorEmbeddings through = RunConnector(
          SafetensorsFile::Open(im.params.dit_path), im.video_connector_cfg,
          im.audio_connector_cfg, prompt_video, prompt_audio, mask, context_tokens);
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
  // ── conditioning on pixels (row LTX25-IMAGE-COND, issue #644) ─────────────
  //
  // Upstream this is `ImageConditioner` (ltx-pipelines/utils/blocks.py:936-993,
  // called at distilled.py:212) feeding `combined_image_conditionings`
  // (utils/helpers.py:272-308). ONE of its four arms is served here, and the
  // other three are refused BY NAME rather than dropped — a keyframe that is
  // silently ignored renders an unconditioned clip that looks like the feature
  // not working.
  //
  // THESE MESSAGES ARE WRITTEN TO BE RE-CHECKABLE, and the count is now SIX
  // refusals in this campaign whose stated reason turned out to be false or
  // stale. Two of the six stood right here. The first said no encoder weights
  // could be materialized — true when written, and what this row fixed. The
  // second replaced it and blamed `keyframes_abs_pos_embedding`, which was
  // verifiably NOT the blocker at the pin (see the last-frame message below for
  // the three anchors that refute it), and a test had been written to assert
  // that wrong reason by name.
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
  if (!gen.ref_image_paths.empty() || !gen.ref_video_dir.empty()) {
    Fail(
        "reference-image / reference-video conditioning is not served. The encoder and the "
        "placement are both here — `Ltx2ConditionVideoByReference` is ported and gated — but "
        "it takes a `downscale_factor` and a `temporal_scale_factor` that must match what the "
        "IC-LoRA was TRAINED with (conditioning/types/reference_video_cond.py:36-37, applied at "
        ":65-77), and "
        "upstream carries those in the LoRA's own metadata, which this project does not read. "
        "A guessed pair places the reference plausibly and wrongly, which no output check can "
        "see, so it is refused instead. Use first_frame_ppm / first_frame_path for "
        "image-to-video.");
  }
  if (!gen.ref_audio_path.empty() || !gen.ref_audio_wav.empty()) {
    Fail(
        "reference-AUDIO conditioning is not served. `Ltx2ConditionAudioByReference` is ported "
        "and gated (conditioning/types/reference_audio_cond.py:34-65), and what it needs is an "
        "encoded waveform: `encode_audio` through the audio VAE's ENCODER "
        "(ltx-pipelines/utils/helpers.py:264-269). This row built the VIDEO encoder's load "
        "path only — there is no AUDIO_VAE_ENCODER key filter — so nothing can turn a WAV into "
        "audio latents here. Recorded as owed.");
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

  // ── geometry ──────────────────────────────────────────────────────────────
  const Ltx2PipelineRecipe& recipe = im.recipe;
  const double fps = recipe.frame_rate;
  const int64_t height = gen.height > 0 ? gen.height : recipe.height;
  const int64_t width = gen.width > 0 ? gen.width : recipe.width;
  int64_t frames = gen.num_frames > 1 ? gen.num_frames : recipe.num_frames;
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
  std::vector<float> audio_latent_volume;  // [C, F, M], unpatchified
  int64_t audio_lc = 0, audio_lf = 0, audio_lm = 0;

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

  for (int64_t phase_index = 0; phase_index <= last_phase; ++phase_index) {
    const Ltx2PhaseRecipe& phase = recipe.phases[static_cast<size_t>(phase_index)];
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

      // `frame_idx == 0` -> `VideoConditionByLatentIndex` (helpers.py:295-300).
      // It REPLACES tokens that already exist and the token count never changes.
      Ltx2LatentState state = ToLatentState(video, /*pos_dims=*/3);
      Ltx2ConditionVideoByLatentIndex(&state, vshape, /*patch_size=*/1, encoded, image_strength,
                                      /*latent_idx=*/0);
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
      const std::vector<float> written(
          video.clean.begin(),
          video.clean.begin() + static_cast<ptrdiff_t>(placed * video.width));
      im.trace.image_tokens = placed;
      im.trace.image_digest = DigestF32(written);
      im.trace.image_absmax = AbsMax(written);
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
      VT_CHECK(video.tokens > target_tokens,
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
      // temporal axis by fps (keyframe_cond.py:52-58), so the first appended
      // token's temporal START is `frame_idx / fps`. Positions are
      // [pos_dims, tokens, 2] concatenated PER DIMENSION, so the temporal axis
      // is dimension 0 and the first appended token sits at `target_tokens * 2`.
      const double want_t0 = static_cast<double>(static_cast<float>(
          static_cast<double>(frames - 1) / fps));
      const double got_t0 = video.positions[static_cast<size_t>(target_tokens * 2)];
      VT_CHECK(std::abs(got_t0 - want_t0) <= 1e-5 * std::max(1.0, std::abs(want_t0)),
               "ltx2 video: the last-frame keyframe's appended tokens must carry the temporal "
               "position of pixel frame `frames - 1` (" +
                   std::to_string(want_t0) + "), but the first appended token starts at " +
                   std::to_string(got_t0) +
                   ". A keyframe that appends the right number of tokens at the wrong TIME "
                   "renders a clip of the right length that pins the image to the wrong end");
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
      const int64_t schedule_tokens = target_tokens;
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

    // ── the denoise loop (samplers.py:39-79 / :488-558) ─────────────────────
    // The ancestral arm's loop generator is seeded from the pipeline seed plus
    // the recipe's own offset (distilled.py:69-73, :177-183) — a separate stream
    // from the state noise, so its first draw is not the initial latent's.
    SplitMixGaussian loop_noise(seed + static_cast<uint64_t>(phase.noise_seed_offset));
    const int64_t sigma_count = static_cast<int64_t>(sigmas.size());
    for (int64_t step = 0; step + 1 < sigma_count; ++step) {
      const float sigma = sigmas[static_cast<size_t>(step)];
      const std::vector<float> v_timesteps = TimestepsFromMask(video, sigma);
      const std::vector<float> a_timesteps = TimestepsFromMask(audio, sigma);
      const float sigma_row = sigma;

      Ltx2ModalityInput vin;
      vin.batch = 1;
      vin.tokens = video.tokens;
      vin.context_tokens = context_tokens;
      vin.latent = video.latent.data();
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
      ain.latent = audio.latent.data();
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

      // One graph, two residencies. On the CPU this is the L2 parity forward in
      // its declared f32; on an accelerator it is the phase-L8 device-resident
      // forward over the bf16 the DiT was STAGED at, and the two agree on
      // everything but where the bytes live and how wide they are.
      const Ltx2DitOutputs velocity =
          im.on_device ? Ltx2DitForwardDevice(*im.queue, im.dit.params, im.dit.weights, &vin,
                                              &ain, im.compute_dtype)
                       : Ltx2DitForward(im.device, im.dit.params, im.dit.weights, &vin, &ain,
                                        im.compute_dtype);

      const std::vector<float> v_denoised = PostProcessLatent(
          ToDenoised(video.latent, velocity.video, v_timesteps, video.tokens, video.width), video);
      const std::vector<float> a_denoised = PostProcessLatent(
          ToDenoised(audio.latent, velocity.audio, a_timesteps, audio.tokens, audio.width), audio);

      const bool terminal = sigmas[static_cast<size_t>(step + 1)] == 0.0F;
      if (phase.stepper == Ltx2StepperKind::kEulerAncestral) {
        if (terminal) {
          // samplers.py:545-547 — the terminal step IS the denoised prediction;
          // taking an ancestral step there would re-noise the finished latent.
          video.latent = v_denoised;
          audio.latent = a_denoised;
          continue;
        }
        const std::vector<float> v_noise =
            loop_noise.Draw(static_cast<int64_t>(video.latent.size()));
        const std::vector<float> a_noise =
            loop_noise.Draw(static_cast<int64_t>(audio.latent.size()));
        video.latent = PostProcessLatent(
            Ltx2EulerAncestralStep(video.latent.data(), v_denoised.data(), sigmas.data(),
                                   sigma_count, step, static_cast<int64_t>(video.latent.size()),
                                   phase.stepper_eta, phase.stepper_s_noise, v_noise.data()),
            video);
        audio.latent = PostProcessLatent(
            Ltx2EulerAncestralStep(audio.latent.data(), a_denoised.data(), sigmas.data(),
                                   sigma_count, step, static_cast<int64_t>(audio.latent.size()),
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
    }

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
    {
      Ltx2LatentState finished = ToLatentState(video, /*pos_dims=*/3);
      Ltx2ClearConditioning(&finished, target_tokens);
      FromLatentState(finished, &video);
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

  int64_t rendered_frames = 0;
  int64_t rendered_channels = 0;
  Ltx2VideoDecodeStreaming(
      im.video_kind, im.video_cfg, im.video_weights, video_latent_volume, video_lc, video_lf,
      video_lh, video_lw, &decode_noise,
      Ltx2AutoTileSizeConfig(rendered_h, rendered_w, video_factors),
      [&](const Ltx2VideoChunk& chunk) {
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
      });

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
    const Ltx2AudioSpectrogram mel = Ltx2AudioDecoderForward(
        im.audio_cfg, im.audio_weights, audio_latent_volume, audio_lc, audio_lf, audio_lm);
    waveform = Ltx2VocoderWithBweForward(im.vocoder_cfg, im.vocoder_weights, mel.data,
                                         mel.channels, mel.frames, mel.mel_bins, &audio_samples);
    audio_channels = mel.channels;
    audio_rate = im.vocoder_cfg.output_sampling_rate;
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
  WriteFileBytes(result.audio_path,
                 MiniMaxH3WriteWav(waveform, audio_channels, audio_samples, audio_rate));
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
  return result;
}

namespace {

std::unique_ptr<VideoEngine> LoadLtx2VideoFamily(const VideoModelParams& params) {
  return Ltx2VideoEngine::Load(params);
}

}  // namespace

REGISTER_VLLM_VIDEO_FAMILY(ltx2, kLtx2VideoFamily, DetectLtx2Video, LoadLtx2VideoFamily)

}  // namespace vllm::multimodal
