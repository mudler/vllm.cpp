// The GENERALIZED video+audio generation seam — the abstract engine every
// diffusion video family is reached through, and the checkpoint-detected
// registry that resolves WHICH family a checkpoint set belongs to.
//
// Why this exists. ARCH-ONE-SURFACE ROW 2 folded MiniMax-H3's whole assembly
// pipeline into ONE library entry point (MiniMaxH3VideoEngine), and the C ABI's
// `vllm_video_*` slice was already shaped generically — separate `dit_path` /
// `encoder_path` / `video_vae_path` / `audio_vae_path` artifacts rather than one
// model directory. Only the INTERNALS were H3-typed. LTX-2.5 is the second
// family (a 21B joint video+audio flow-matching DiT), and AGENTS.md §"Shared
// seams" is explicit that new models are ADDITIVE files reached through the
// shared surface — never a second parallel path. So the H3-typed entry point
// becomes one implementation of this interface, and a new family is a new file
// plus one REGISTER_VLLM_VIDEO_FAMILY line.
// Spec: .agents/specs/ltx-2-5.md §5. Issue #435.
//
// FAMILY-SPECIFIC FIELDS DO NOT LAND HERE. H3's `partition` (fl2va|ref2va) and
// LTX's `pipeline_kind` / `model_version` ride in `extras`, a string map, so
// adding a family adds no member to a struct every other family must then
// ignore — the same reason vLLM keeps per-architecture knobs in the config dict
// rather than in EngineArgs.
//
// NEVER GUESS A FAMILY. `LoadVideoEngine` either honours an explicitly declared
// `family`, or asks every registered family to look at the checkpoint. Exactly
// one claimant loads; zero or several is a std::runtime_error naming what was
// seen and what is registered. "There is only one family registered, so it must
// be that one" is precisely the silent mis-load this seam exists to prevent —
// an H3 GGUF handed to an LTX loader does not fail, it renders noise.
#pragma once

#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "vt/device.h"

namespace vllm::openai {
struct VideoRequest;  // entrypoints/openai/video_api.h
}

namespace vllm::multimodal {

// ── Load-time parameters: the checkpoint SET. Empty string == "not supplied".
// The C ABI mirror is vllm_video_model_params. ───────────────────────────────
struct VideoModelParams {
  std::string dit_path;       // the denoiser: GGUF | safetensors | shard DIR
  std::string encoder_path;   // the text tower, when the family has one
  std::string tokenizer_path; // tokenizer.json (prompt conditioning)
  std::string video_vae_path, video_vae_config_path;
  std::string audio_vae_path, audio_vae_config_path;
  // Fallback conditioning when no encoder is configured: rows of the family's
  // text width, little-endian f32.
  std::string prompt_embeds_path;

  // The family to load. EMPTY means "detect from the checkpoint"; a non-empty
  // value that is not registered is refused (it is never treated as a hint).
  std::string family;

  // 0 is the CPU; 1 is the accelerator this build RESOLVES through the platform
  // seam, not the enum value 1 (#659, #660). Refused by name when this build has
  // no accelerator backend, or its partial backend declines the family.
  int32_t device = 0;
  int32_t dequant_bf16 = 0;  // 0 keep-quant, 1 dequant/stream bf16
  int32_t fp4_resident = 0;  // keep packed FP4 resident + quantized GEMM
  int64_t encoder_max_layers = 0;  // 0 => all layers

  // Family-specific load knobs. H3: "partition" = "fl2va" | "ref2va".
  std::map<std::string, std::string> extras;
};

// ── Per-generation parameters (the C ABI mirror is vllm_video_params). ───────
struct VideoGenParams {
  std::string prompt;  // encoded when the engine has an encoder
  // "" => resolved by the family from the references it was given.
  std::string task;
  double duration_seconds = 0.0;  // <= 0 => per-task default
  int64_t num_frames = 0;         // <= 1 => per-task default
  int64_t height = 0, width = 0;  // <= 0 => aspect-derived default
  int64_t steps = 0;              // <= 0 => the family's default
  double flow_shift = 0.0;        // <= 0 => the family's default
  double audio_flow_shift = 0.0;  // <= 0 => the family's default
  uint64_t seed = 0;
  bool has_seed = false;  // false => the family's fixed default noise streams

  // KEYFRAMES: binary PPM (P6), as a path or in-memory bytes (exactly one
  // spelling per frame). Pin frame 0 / the last frame OF THE OUTPUT.
  std::string first_frame_path, last_frame_path;
  std::string first_frame_ppm;  // in-memory alternative (server data: URLs)
  double noise_aug = 1.0;       // condition-noise augmentation (1.0 pins)

  // REFERENCES (exclusive with keyframes).
  std::vector<std::string> ref_image_paths;  // whole reference images (PPM)
  std::string ref_video_dir;                 // DIR of frame_%06d.ppm
  std::string ref_audio_path;                // 16-bit PCM WAV path...
  std::string ref_audio_wav;                 // ...or its bytes

  // Where frame_%06d.ppm + audio.wav land (created if absent). REQUIRED.
  std::string output_dir;

  // Family-specific per-request knobs (none for H3 today).
  std::map<std::string, std::string> extras;
};

// ── One finished generation (the C ABI mirror is vllm_video_result). ────────
struct VideoResult {
  std::string frame_dir;   // holds frame_%06d.ppm
  std::string audio_path;  // 16-bit PCM WAV
  int64_t frame_count = 0, width = 0, height = 0;
  int64_t fps = 0, sample_rate = 0;
  // The ffmpeg argv the CALLER may exec (argv[0] is "ffmpeg"). THE PROCESS
  // BOUNDARY is part of this seam's contract, inherited from ROW 2: the library
  // writes artifacts and COMPOSES the argv, and spawns nothing.
  std::vector<std::string> mux_argv;
  std::string mux_output_path;  // the -o target mux_argv names
};

// Read a family-specific extra, or `fallback` when the key is absent.
std::string VideoExtra(const std::map<std::string, std::string>& extras, const std::string& key,
                       const std::string& fallback = std::string());

// Every tensor name a checkpoint artifact DECLARES, header-only — no payload is
// read, so this is safe on a checkpoint far larger than RAM. `path` may be a
// single GGUF file (including a llama.cpp split set), a single safetensors
// file, or a multi-shard directory (read through its index's weight_map, the
// same two index spellings the shard loaders accept). Returns false with *why
// holding the reason when the artifact cannot be enumerated.
//
// This is the ONE thing every family detector inspects, so that detection is a
// question about what a checkpoint HOLDS rather than what its filename says,
// and so two families cannot disagree about how a checkpoint was read.
bool ReadVideoCheckpointTensorNames(const std::string& path, std::vector<std::string>* out,
                                    std::string* why);

// A loaded video checkpoint set, weights staged once, ready to generate.
class VideoEngine {
 public:
  virtual ~VideoEngine();

  // The stable registry name of the family this engine implements. Stable
  // because it is what a caller DECLARES to skip detection and what an
  // unresolved load prints.
  virtual std::string family() const = 0;

  // The device the queue created during load selected.
  virtual vt::Device device() const = 0;

  // True when a text tower is loaded (the request PROMPT conditions the
  // render); false => prompt-embeds conditioning (or Generate refuses).
  virtual bool has_encoder() const = 0;
  virtual bool has_prompt_embeds() const = 0;

  // Run one blocking generation. Implementations serialize internally (staged
  // weights are shared state); throws std::runtime_error to fail the request.
  virtual VideoResult Generate(const VideoGenParams& params) = 0;

 protected:
  // Polymorphic base: constructible and movable only by derived classes.
  VideoEngine() = default;
  VideoEngine(const VideoEngine&) = default;
  VideoEngine& operator=(const VideoEngine&) = default;
  VideoEngine(VideoEngine&&) = default;
  VideoEngine& operator=(VideoEngine&&) = default;
};

// ── The registry ────────────────────────────────────────────────────────────
// Mirrors the ModelRegistry self-registration idiom (model_registry.h): each
// family registers itself from its OWN translation unit, so adding a family
// edits no shared array.

// Does this checkpoint set belong to the family? Implementations INSPECT the
// artifact (tensor names, metadata) — never the file extension or the path
// spelling, both of which are chosen by whoever repackaged the checkpoint. A
// detector must not throw: an unreadable or unrecognizable artifact is `false`,
// and the caller reports it with the rest of the evidence.
using VideoFamilyDetector = std::function<bool(const VideoModelParams&)>;

// Load the checkpoint set as this family. Throws std::runtime_error naming the
// problem on any mismatch.
using VideoFamilyLoader = std::function<std::unique_ptr<VideoEngine>(const VideoModelParams&)>;

struct VideoFamilyRegistration {
  std::string name;  // the stable family name, e.g. "minimax-h3"
  VideoFamilyDetector detect;
  VideoFamilyLoader load;
};

// Add a family to the process-global registry. Throws std::runtime_error on an
// empty name, a missing detector or loader, or A NAME ALREADY REGISTERED — the
// last because two families sharing one name is the never-guess guarantee
// defeated from the inside: the listing shows one family, two claimants collapse
// into one name so the SEVERAL-claimants refusal cannot fire, and which loader
// runs falls to link order. Registrars run at static init, so a refusal there
// ends the process; that is intended, since a name collision is a build defect
// and a checkpoint handed to the wrong family renders noise rather than failing.
void RegisterVideoFamily(VideoFamilyRegistration registration);

// Every registered family name, sorted and duplicate-free — invariants of the
// registry itself (RegisterVideoFamily inserts in order and refuses a
// collision), so they hold however static init ordered the TUs and however late
// a caller registers. This listing is what refusals print.
std::vector<std::string> RegisteredVideoFamilies();

// Every registered family that CLAIMS this checkpoint set, sorted. Empty means
// nothing recognized it; more than one means the detectors overlap, which is a
// defect in whichever detector is too loose — resolved by refusing, never by
// picking one.
std::vector<std::string> DetectVideoFamilies(const VideoModelParams& params);

// Resolve the family and load. `params.family`, when non-empty, selects
// directly (an unregistered name is refused naming what IS registered). When
// empty, detection must produce exactly one claimant; zero or several throws
// std::runtime_error naming the checkpoint it inspected, what it found there,
// and every registered family.
std::unique_ptr<VideoEngine> LoadVideoEngine(const VideoModelParams& params);

// The ONE mapping from a parsed /v1/videos request onto the seam's params —
// library-owned so the HTTP route and the FFI cannot drift, and family-agnostic
// so a second family serves the same endpoint without a second mapping.
// `output_dir` is the job directory the artifacts land in.
VideoGenParams VideoGenParamsFromRequest(const ::vllm::openai::VideoRequest& request,
                                         const std::string& output_dir);

// Static-init helper whose constructor performs the self-registration; used
// only through the REGISTER_VLLM_VIDEO_FAMILY macro.
struct VideoFamilyRegistrar {
  explicit VideoFamilyRegistrar(VideoFamilyRegistration registration) {
    RegisterVideoFamily(std::move(registration));
  }
};

// Registers one family from its own TU. Place at namespace scope inside
// `namespace vllm::multimodal { ... }`; `unique_tag` is any TU-unique token.
#define REGISTER_VLLM_VIDEO_FAMILY(unique_tag, family_name, detect_fn, load_fn)      \
  namespace {                                                                        \
  const ::vllm::multimodal::VideoFamilyRegistrar vllm_video_family_registrar_##unique_tag( \
      ::vllm::multimodal::VideoFamilyRegistration{(family_name), (detect_fn), (load_fn)}); \
  }  // namespace

}  // namespace vllm::multimodal
