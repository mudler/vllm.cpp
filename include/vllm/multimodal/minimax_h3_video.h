// MiniMax-H3 video+audio generation seam — the ONE library entry point that
// composes checkpoint loading (DiT GGUF/NVFP4/sharded-bf16, both VAEs, the
// optional H3-Encoder text tower) -> task resolution + the #77 partition guard
// -> conditioning (prompt encode / prompt-embeds file / fl2va keyframes /
// ref2va references) -> the deterministic noise draw -> MiniMaxH3GenerateT2va
// -> artifacts (frame_%06d.ppm + audio.wav) + the ffmpeg argv, and that every
// consumer (C ABI `vllm_video_*`, the OpenAI server's /v1/videos, the
// `minimax-h3-gen` example) drives.
//
// ARCH-ONE-SURFACE ROW 2 (video+audio generation). This file ABSORBS the
// assembly pipeline `examples/minimax_h3_gen/main.cpp` owned privately
// (pre-refactor main.cpp:687-1288 @ fc636c76: loader-arm dispatch, encoder
// conditioning, reference encoding, the splitmix64 noise streams, artifact
// writing, mux-argv assembly) and the twin copy `examples/server/main.cpp`
// carried for /v1/videos (pre-refactor main.cpp:743-1096). Per the ONE SURFACE
// directive the examples keep argv parsing, printing and the PROCESS SPAWN
// only; the capability lives here, reachable by any embedder.
//
// THE PROCESS BOUNDARY (developer-ratified 2026-08-03) stands: this seam
// WRITES artifacts and BUILDS the mux argv (MiniMaxH3BuildMp4MuxArgs) and
// spawns NOTHING — the caller execs `MiniMaxH3VideoResult::mux_argv`.
//
// Upstream mirror shape: vLLM-Omni serves H3 through
// vllm_omni/diffusion/models/minimax_h3/pipeline_minimax_h3.py (the pipeline
// object owns checkpoints + does per-request _resolve_task/_resolve_shape);
// this seam is the C++ engine-side equivalent of that pipeline object.
//
// Byte-identity contract: on the committed fold fixture the CPU t2va render is
// byte-identical to the PRE-fold `minimax-h3-gen` binary at fc636c76
// (tests/vllm/models/test_minimax_h3_video_fold.cpp, three-arm gate).
//
// LTX-2.5 L1 (.agents/specs/ltx-2-5.md §5, issue #435): H3 is now ONE FAMILY
// behind the generalized `vllm::multimodal::VideoEngine` seam, registered as
// "minimax-h3" and reached through `LoadVideoEngine`. Nothing below changed
// behaviourally — the H3-typed params/result structs, `Load`, and the H3-typed
// `Generate` are the same entry points on the same code path, and the fold
// gate above still holds them to the same golden bytes. What is new is that the
// same handle also satisfies the abstract seam, and that the H3-specific
// `partition` field maps to/from the generic `extras["partition"]`.
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "vllm/multimodal/video_engine.h"
#include "vt/device.h"

namespace vllm::openai {
struct VideoRequest;  // entrypoints/openai/video_api.h
}

namespace vllm::multimodal {

// The stable registry name this family is reached under (VideoModelParams::family
// / vllm_video_model_params.family).
inline constexpr char kMiniMaxH3VideoFamily[] = "minimax-h3";

// Map the stable public video ABI device selector onto the runtime's generic
// backend key. The ABI is unchanged — 0 is the CPU, 1 is the accelerator, and
// anything else throws — but 1 is RESOLVED through the platform seam
// (`CurrentPlatform().device_type()` + `TryGetBackend` +
// `supports_model_architecture`), never cast from the integer. It therefore
// THROWS on a build with no accelerator backend, or one whose partial backend
// declines this architecture, instead of naming a device that build cannot run
// (#659, #660). Callers below this seam dispatch only through the returned
// DeviceType.
vt::DeviceType MiniMaxH3VideoDeviceType(int32_t device);

// ── Load-time parameters (the checkpoint set; the C ABI mirror is
// vllm_video_model_params). Empty string == "not supplied". ─────────────────
struct MiniMaxH3VideoModelParams {
  std::string dit_path;            // GGUF | NVFP4 safetensors | bf16 shard DIR
  std::string encoder_path;        // H3-Encoder GGUF or bf16 shard DIR
  std::string tokenizer_path;      // tokenizer.json (encoder conditioning)
  std::string video_vae_path, video_vae_config_path;
  std::string audio_vae_path, audio_vae_config_path;
  // Fallback conditioning when no encoder is configured: rows of text_dim,
  // little-endian f32 (the pre-fold --prompt-embeds / --video-prompt-embeds).
  std::string prompt_embeds_path;
  // The served checkpoint PARTITION ("fl2va" | "ref2va"). Community GGUF/NVFP4
  // files strip the release model_index.json `_minimax_h3` block and the two
  // DiTs are byte-structurally identical, so it must be DECLARED; empty is
  // declared-but-unknown and the #77 guard refuses every full render
  // (MiniMaxH3PartitionFromFlag / MiniMaxH3CheckTaskPartition).
  std::string partition;
  // 0 cpu, 1 the accelerator this build resolves — see MiniMaxH3VideoDeviceType
  // above; never `static_cast<vt::DeviceType>(device)` (#660).
  int32_t device = 0;
  int32_t dequant_bf16 = 0;  // 0 keep-quant, 1 dequant/stream bf16
  // NVFP4 + cuda only: keep the packed FP4 resident and route the quantized
  // projections through the Marlin W4A16 GEMM (the pre-fold --fp4-resident).
  int32_t fp4_resident = 0;
  int64_t encoder_max_layers = 0;  // 0 => all layers
};

// ── Per-generation parameters (the C ABI mirror is vllm_video_params). ──────
struct MiniMaxH3VideoGenParams {
  std::string prompt;  // encoded when the engine has an encoder
  // "" => resolved from the references/partition (upstream _resolve_task):
  // a keyframe => fl2va, any ref2va reference => ref2va, else t2va.
  std::string task;
  double duration_seconds = 0.0;         // <= 0 => per-task default
  int64_t num_frames = 0;                // <= 1 => per-task default
  int64_t height = 0, width = 0;         // <= 0 => aspect-derived default
  int64_t steps = 0;                     // <= 0 => H3 default (50)
  double flow_shift = 0.0;               // <= 0 => H3 default (12.0)
  double audio_flow_shift = 0.0;         // <= 0 => H3 default (3.0)
  uint64_t seed = 0;
  bool has_seed = false;  // false => the pre-fold fixed default streams

  // fl2va KEYFRAMES: binary PPM (P6), as a path or in-memory bytes (exactly
  // one spelling per frame). Pins frame 0 / the last frame OF THE OUTPUT.
  std::string first_frame_path, last_frame_path;
  std::string first_frame_ppm;  // in-memory alternative (server data: URLs)
  double noise_aug = 1.0;       // condition-noise augmentation (1.0 pins)

  // ref2va REFERENCES (exclusive with keyframes, minimax_h3_pipeline.cpp:251):
  std::vector<std::string> ref_image_paths;  // whole reference images (PPM)
  std::string ref_video_dir;                 // DIR of frame_%06d.ppm
  std::string ref_audio_path;                // 16-bit PCM WAV path...
  std::string ref_audio_wav;                 // ...or its bytes

  // Where frame_%06d.ppm + audio.wav land (created if absent). REQUIRED.
  std::string output_dir;
};

// ── One finished generation (the C ABI mirror is vllm_video_result). ────────
struct MiniMaxH3VideoResult {
  std::string frame_dir;   // holds frame_%06d.ppm
  std::string audio_path;  // 16-bit PCM WAV
  int64_t frame_count = 0, width = 0, height = 0;
  int64_t fps = 0, sample_rate = 0;
  // The ffmpeg argv the CALLER may exec to mux <output_dir>/video.mp4
  // (argv[0] is "ffmpeg"; substitute a custom binary before exec'ing).
  std::vector<std::string> mux_argv;
  std::string mux_output_path;  // the -o target mux_argv names
};

// A loaded H3 video checkpoint set, weights staged once, ready to generate.
// Construction throws std::runtime_error naming the problem on any mismatch.
//
// It IS a VideoEngine (L1): the generic overrides below are thin adapters over
// the H3-typed members — same Impl, same forward, same bytes.
class MiniMaxH3VideoEngine final : public VideoEngine {
 public:
  static std::unique_ptr<MiniMaxH3VideoEngine> Load(const MiniMaxH3VideoModelParams& params);

  MiniMaxH3VideoEngine(MiniMaxH3VideoEngine&&) noexcept;
  MiniMaxH3VideoEngine& operator=(MiniMaxH3VideoEngine&&) noexcept;
  ~MiniMaxH3VideoEngine() override;

  // Always kMiniMaxH3VideoFamily.
  std::string family() const override;

  // The device selected by the queue created during Load().
  vt::Device device() const override;

  // True when an encoder tower is loaded (the request PROMPT conditions the
  // render); false => prompt_embeds_path conditioning (or Generate refuses).
  bool has_encoder() const override;
  bool has_prompt_embeds() const override;

  // Run one blocking generation. Serialized internally (the staged weights are
  // shared state); throws std::runtime_error to fail the request.
  MiniMaxH3VideoResult Generate(const MiniMaxH3VideoGenParams& params);

  // The generic spelling: converts, calls the H3-typed Generate above, converts
  // back. There is no second code path.
  VideoResult Generate(const VideoGenParams& params) override;

 private:
  MiniMaxH3VideoEngine();
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

// ── The generic <-> H3 param mapping (single-sourced so the C ABI, the HTTP
// route and the registry loader cannot drift). `partition` is the ONE
// H3-specific load field, and it rides in the generic `extras` under the key
// "partition" — absent means declared-but-unknown, exactly as an empty
// MiniMaxH3VideoModelParams::partition does (the #77 guard). ───────────────────
MiniMaxH3VideoModelParams MiniMaxH3VideoModelParamsFromGeneric(const VideoModelParams& params);
VideoModelParams MiniMaxH3VideoModelParamsToGeneric(const MiniMaxH3VideoModelParams& params);
MiniMaxH3VideoGenParams MiniMaxH3VideoGenParamsFromGeneric(const VideoGenParams& params);
VideoResult MiniMaxH3VideoResultToGeneric(const MiniMaxH3VideoResult& result);

// The ONE mapping from a parsed /v1/videos request onto the seam's params —
// library-owned so the HTTP route and the FFI cannot drift (the pre-fold
// server carried this as a private lambda). `output_dir` is the job directory
// the artifacts land in.
MiniMaxH3VideoGenParams MiniMaxH3VideoGenParamsFromRequest(
    const ::vllm::openai::VideoRequest& request, const std::string& output_dir);

}  // namespace vllm::multimodal
